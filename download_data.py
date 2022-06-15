"""Code taken from
https://github.com/gwastro/ml-mock-data-challenge-1/blob/master/generate_data.py
"""

from argparse import ArgumentParser
import os
import requests
import tqdm


def base_path():
    return os.path.split(os.path.abspath(__file__))[0]


def get_default_path():
    return os.path.join(base_path(), 'real_noise_file.hdf')


def download_data(path=None, resume=True):
    """Download noise data from the central server.
    
    Arguments
    ---------
    path : {str or None, None}
        Path at which to store the file. Must end in `.hdf`. If set to
        None a default path will be used.
    resume : {bool, True}
        Resume the file download if it was interrupted.
    """
    if path is None:
        path = get_default_path()
    assert os.path.splitext(path)[1] == '.hdf'
    url = 'https://www.atlas.aei.uni-hannover.de/work/marlin.schaefer/MDC/real_noise_file.hdf'  # noqa: E501
    header = {}
    resume_size = 0
    if os.path.isfile(path) and resume:
        mode = 'ab'
        resume_size = os.path.getsize(path)
        header['Range'] = f'bytes={resume_size}-'
    else:
        mode = 'wb'
    with open(path, mode) as fp:
        response = requests.get(url, stream=True, headers=header)
        total_size = response.headers.get('content-length')

        if total_size is None:
            print("No file length found")
            fp.write(response.content)
        else:
            total_size = int(total_size)
            desc = f"Downloading real_noise_file.hdf to {path}"
            print(desc)
            with tqdm.tqdm(total=int(total_size),
                           unit='B',
                           unit_scale=True,
                           dynamic_ncols=True,
                           desc="Progress: ",
                           initial=resume_size) as progbar:
                for data in response.iter_content(chunk_size=4000):
                    fp.write(data)
                    progbar.update(4000)


def main():
    parser = ArgumentParser()
    
    parser.add_argument('--path', type=str,
                        help="Path to store the file at. "
                             "Default: real_noise_file.hdf")
    parser.add_argument('--no-resume', type='store_true',
                        help="Restart the download rather than resuming it.")
    
    args = parser.parse_args()
    
    download_data(path=args.path, resume=not args.no_resume)
    return


if __name__ == "__main__":
    main()
