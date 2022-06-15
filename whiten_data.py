from argparse import ArgumentParser
import h5py
from pycbc.types import TimeSeries
from pycbc import DYN_RANGE_FAC
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
import time
import queue


def worker(pidx, fpath, inpipe, output, event):
    key = None
    while not event.is_set():
        try:
            key = inpipe.get(timeout=0.01)
        except queue.Empty:
            pass
        if key is None:
            time.sleep(0.1)
            continue
        with h5py.File(fpath, 'r') as fp:
            for det in ['H1', 'L1']:
                ds = fp[f'{det}/{key}']
                tmp = ds[()]
                data = TimeSeries(tmp, epoch=ds.attrs['start_time'],
                                  delta_t=ds.attrs['delta_t'])
                data = data.astype(np.float64) * DYN_RANGE_FAC
                data = data.whiten(4, 4)
                
                posted = False
                while not posted:
                    try:
                        output.send((data.numpy(),
                                     data.delta_t,
                                     float(data.start_time),
                                     det,
                                     str(int(key) - 4)))
                        posted = True
                    except queue.Full:
                        continue
        key = None


def main():
    parser = ArgumentParser()
    
    parser.add_argument('--file', type=str,
                        help="The file that should be whitened.")
    parser.add_argument('--output', type=str,
                        help=("Path to the file at which to store the "
                              "whitened data."))
    parser.add_argument('--workers', type=int, default=0,
                        help=("The number of processes to use for whitening. "
                              "If set to something smaller 1, the number of "
                              "CPUs will be used. Default: 0"))
    parser.add_argument('--force', action='store_true',
                        help="Overwrite existing files.")
    
    args = parser.parse_args()
    
    if args.workers < 1:
        args.workers = mp.cpu_count()
    
    with h5py.File(args.file, 'r') as fp:
        keys = sorted(fp['H1'].keys(), key=lambda inp: int(inp))
    
    keyqueue = mp.Queue()
    for key in keys:
        keyqueue.put(key)
    for _ in range(10):
        key = keys.pop(0)
        keyqueue.put(key)
    event = mp.Event()
    
    processes = []
    sendpipes = []
    recvpipes = []
    for pidx in range(args.workers):
        sendpipe, recvpipe = mp.Pipe()
        sendpipes.append(sendpipe)
        recvpipes.append(recvpipe)
        p = mp.Process(target=worker,
                       args=(pidx,
                             args.file,
                             keyqueue,
                             sendpipe,
                             event))
        processes.append(p)
        p.start()
    
    pbar = tqdm(ascii=True, total=len(keys) * 2)
    mode = 'w' if args.force else 'x'
    with pbar as bar, h5py.File(args.output, mode) as fp:
        while True:
            for pipe in recvpipes:
                try:
                    while pipe.poll(timeout=0.01):
                        data, dt, st, det, key = pipe.recv()
                        if det not in fp:
                            fp.create_group(det)
                            fp[det].create_dataset(key, data=data)
                        fp[det][key].attrs['start_time'] = st
                        fp[det][key].attrs['delta_t'] = dt
                        try:
                            sendkey = keys.pop(0)
                            keyqueue.put(sendkey)
                        except IndexError:
                            pass
                        bar.update(1)
                except queue.Empty:
                    continue
            
            if bar.n == 2 * len(keys):
                event.set()
                while True:
                    try:
                        queue.get(timeout=0.01)
                    except queue.Empty:
                        break
                for pipe in sendpipes:
                    while pipe.poll():
                        pipe.recv()
                for pipe in recvpipes:
                    while pipe.poll():
                        pipe.recv()
                for p in processes:
                    p.join()
    return


if __name__ == "__main__":
    main()
