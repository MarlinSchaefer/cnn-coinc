import h5py
import numpy as np
import os
from tensorflow import keras
from argparse import ArgumentParser

from BnsLib.network import H5pyHandler, FileHandler, MultiFileHandler,\
                           PrefetchedFileGeneratorMP
from BnsLib.data import number_segments
from BnsLib.types import MultiArrayIndexer
from BnsLib.utils import inverse_string_format


class NoiseHandler(H5pyHandler):
    def __init__(self, *args, window_size=2048, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_size = window_size
        with h5py.File(self.file_path, 'r') as fp:
            self.calculate_lengths(fp)
    
    def __len__(self):
        return len(self.indexer)
    
    def calculate_lengths(self, fp):
        self.indexer = MultiArrayIndexer()
        for key in sorted(fp['H1'].keys(), key=lambda inp: int(inp)):
            ds = fp[f'H1/{key}']
            nsegs = number_segments(len(ds),
                                    self.window_size,
                                    self.window_size)
            self.indexer.add_length(nsegs, name=key)
    
    def _getitem_open(self, index, fp):
        key, idx = list(self.indexer[int(index)].items())[0]
        sidx = idx * self.window_size
        eidx = sidx + self.window_size
        data = fp[f'H1/{key}'][sidx:eidx]
        return np.expand_dims(data, axis=-1)
    
    def serialize(self):
        dic = super().serialize()
        dic.update({'window_size': self.window_size})
        return dic


class SignalHandler(H5pyHandler):
    def __init__(self, *args, nsamples=2048, **kwargs):
        super().__init__(*args, **kwargs)
        self.nsamples = nsamples
    
    def __len__(self):
        if self.file is None:
            with h5py.File(self.file_path, 'r') as fp:
                length = len(fp['data/0'])
        else:
            length = len(self.file['data/0'])
        return length
    
    def _getitem_open(self, index, fp):
        snr = np.random.randint(5, 15)
        data = fp['data/0'][index]
        nsamples = min(len(data), self.nsamples)
        label = np.array([1, 0])
        return snr * np.expand_dims(data[-nsamples:], axis=-1), label
    
    def serialize(self):
        dic = super().serialize()
        dic.update({'nsamples': self.nsamples})
        return dic


class NoSignalHandler(FileHandler):
    def __init__(self, shape=(2048, 1)):
        super().__init__(None)
        self.shape = shape
    
    def __contains__(self, index):
        return index == -1
    
    def __len__(self):
        return 1
    
    def open(self, mode='r'):
        return
    
    def close(self):
        return
    
    def __enter__(self):
        return
    
    def __exit__(self, exc_type, exc_code, exc_traceback):
        return
    
    def __getitem__(self, index):
        return np.zeros(self.shape), np.array([0, 1])
    
    def serialize(self):
        dic = {'shape': self.shape}
        return dic
    
    @classmethod
    def from_serialized(cls, dic):
        return cls(shape=dic['shape'])


class MultiHandler(MultiFileHandler):
    @classmethod
    def from_serialized(cls, dic):
        handlers = [NoiseHandler, SignalHandler, NoSignalHandler]
        return super().from_serialized(dic, handlers)
    
    def split_index_to_groups(self, index):
        nidx, sidx = index
        return {'noise': nidx, 'signal': sidx}
    
    def format_return(self, inp):
        signal, label = inp['signal']
        noise = inp['noise']
        return signal + noise, label


def get_generator(signal_files, noise_file, batch_size=32, shuffle=True,
                  nsig=None, noirange=None, seed=None, noise_per_signal=None,
                  ratio=1, prefetch=10, workers=10):
    # Setting up file-handlers
    mh = MultiHandler()
    mh.input_shape = (4 * 2048, 1)
    mh.output_shape = (1, 1)
    nosig = NoSignalHandler(shape=mh.input_shape)
    mh.add_file_handler(nosig, group="signal")
    
    # Setting up signal handlers
    if not isinstance(signal_files, list):
        signal_files = [signal_files]
    total_sigs = 0
    for sigfile in signal_files:
        fh = SignalHandler(sigfile, base_index=total_sigs,
                           nsamples=mh.input_shape[0])
        length = len(fh)
        total_sigs += length
        mh.add_file_handler(fh, group="signal")
    
    # Setting up noise handlers
    nh = NoiseHandler(noise_file, window_size=mh.input_shape[0])
    mh.add_file_handler(nh, group="noise")
    
    # Generate index list
    if nsig is None:
        nsig = total_sigs
    sigidxs = np.array(list(range(nsig)), dtype=int)
    
    if noirange is None:
        noirange = (0, len(nh))
    elif isinstance(noirange, int):
        noirange = (0, noirange)
    noiidxs = np.array(list(range(*noirange)), dtype=int)
    
    rs = np.random.RandomState(seed)
    index_list = []
    if noise_per_signal is None:
        noise_per_signal = 1
    for i in range(noise_per_signal):
        nidxs = rs.randint(0, len(noiidxs), size=len(sigidxs))
        index_list.extend(list(np.stack([noiidxs[nidxs], sigidxs]).T))
    
    num_pure_noise = int(len(index_list) / ratio)
    pure_noise_indices = np.random.choice(np.arange(len(nh)),
                                          size=num_pure_noise)
    index_list.extend([[pni, -1] for pni in pure_noise_indices])
    
    # Instantiate generator
    generator = PrefetchedFileGeneratorMP(mh, index_list,
                                          batch_size=batch_size,
                                          shuffle=shuffle,
                                          prefetch=prefetch,
                                          workers=workers)
    
    return generator


def get_network():
    inp = keras.layers.Input((4 * 2048, 1))
    b1 = keras.layers.BatchNormalization()(inp)
    c1 = keras.layers.Conv1D(8, 64)(b1)
    a1 = keras.layers.Activation('elu')(c1)
    c2 = keras.layers.Conv1D(8, 32)(a1)
    p1 = keras.layers.MaxPooling1D(8)(c2)
    a2 = keras.layers.Activation('elu')(p1)
    c3 = keras.layers.Conv1D(16, 32)(a2)
    a3 = keras.layers.Activation('elu')(c3)
    c4 = keras.layers.Conv1D(16, 16)(a3)
    p2 = keras.layers.MaxPooling1D(6)(c4)
    a4 = keras.layers.Activation('elu')(p2)
    c5 = keras.layers.Conv1D(32, 16)(a4)
    a5 = keras.layers.Activation('elu')(c5)
    c6 = keras.layers.Conv1D(32, 16)(a5)
    p3 = keras.layers.MaxPooling1D(4)(c6)
    a6 = keras.layers.Activation('elu')(p3)
    f1 = keras.layers.Flatten()(a6)
    d1 = keras.layers.Dense(64)(f1)
    dr1 = keras.layers.Dropout(0.5)(d1)
    a7 = keras.layers.Activation('elu')(dr1)
    d2 = keras.layers.Dense(64)(a7)
    dr2 = keras.layers.Dropout(0.5)(d2)
    a8 = keras.layers.Activation('elu')(dr2)
    out = keras.layers.Dense(2, activation='softmax')(a8)
    model = keras.models.Model(inputs=[inp], outputs=[out])
    return model


def main():
    parser = ArgumentParser()
    
    parser.add_argument('--datadir', type=str, required=True,
                        help="Path at which to find the training data files.")
    parser.add_argument('--outdir', type=str, required=True,
                        help="Path to directory in which to store the output.")
    
    args = parser.parse_args()
    model = get_network()
    
    opti = keras.optimizers.Adam(learning_rate=0.002, beta_1=0.9,
                                 beta_2=0.999, epsilon=1e-8)
    model.compile(loss='binary_crossentropy', metrics='acc',
                  optimizer=opti)
    
    form = 'signals-{index}.hdf'
    sigfiles = list(filter(lambda fn: inverse_string_format(fn, form) is not None,  # noqa: E501
                           os.listdir(args.datadir)))
    sigfiles = [os.path.join(args.datadir, fn) for fn in sigfiles]
    valsplit = 0.8
    valsplitidx = int(len(sigfiles) * valsplit)
    trsigfiles = sigfiles[:valsplitidx]
    valsigfiles = sigfiles[valsplitidx:]
    noifile = os.path.join(args.datadir, 'noise.hdf')
    gen = get_generator(trsigfiles, noifile,
                        noirange=(0, valsplitidx * 20_000))
    valgen = get_generator(valsigfiles, noifile,
                           noirange=(valsplitidx * 20_000,
                                     valsplitidx * 40_000))
    
    csvpath = os.path.join(args.outdir, 'history.csv')
    csvlogger = keras.callbacks.CSVLogger(csvpath)
    
    ckptpath = os.path.join(args.outdir, 'model_{epoch}')
    ckpt = keras.callbacks.ModelCheckpoint(ckptpath, save_best_only=False,
                                           save_weights_only=False,
                                           save_freq='epoch')
    
    with gen, valgen:
        model.fit(gen, validation_data=valgen, shuffle=False,
                  workers=0, use_multiprocessing=False,
                  callbacks=[csvlogger, ckpt], epochs=100)
    return


if __name__ == "__main__":
    main()
