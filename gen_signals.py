from argparse import ArgumentParser
from subprocess import run
from urllib.request import urlretrieve
import os
import numpy as np


def main():
    parser = ArgumentParser()
    
    parser.add_argument('-n', '--nsignals', type=int, required=True,
                        help="The number of signals to generate.")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="Path at which to store the output file.")
    parser.add_argument('-s', '--seed', type=int,
                        help="The seed used to create the injections. "
                             "Default: Random seed")
    parser.add_argument('--verbose', action='store_true',
                        help="Print status updates.")
    parser.add_argument('--force', action='store_true',
                        help="Overwrite existing files.")
    
    args = parser.parse_args()
    
    if args.seed is None:
        args.seed = np.random.randint(0, 2**32 - 1)
    
    # Download psd-file from MLGWSC-1 repository
    if not os.path.isfile('psd-0.hdf') or args.force:
        psdurl = ('https://github.com/gwastro/ml-mock-data-challenge-1/blob/'
                  'master/psds/H1/psd-0.hdf?raw=true')
        urlretrieve(psdurl, 'psd-0.hdf')
    
    # Download ds4.ini file from MLGWSC-1 repo
    if not os.path.isfile('injections.ini') or args.force:
        iniurl = ('https://raw.githubusercontent.com/gwastro/'
                  'ml-mock-data-challenge-1/master/ds4.ini')
        urlretrieve(iniurl, 'injections.ini')
    
    # Run pycbc_create_injections to draw signal parameters
    cmd = ['pycbc_create_injections']
    cmd += ['--config-files', 'injections.ini']
    cmd += ['--ninjections', str(args.nsignals)]
    cmd += ['--seed', str(args.seed)]
    cmd += ['--output-file', 'injections.hdf']
    cmd += ['--force']
    if args.verbose:
        cmd += ['--verbose']
    run(cmd)
    
    # Run the signal generation script from BnsLib
    cmd = ['bnslib_generate_samples']
    cmd += ['--parameters', 'injections.hdf']
    cmd += ['--approximant', 'IMRPhenomXPHM']
    cmd += ['--detectors', 'H1']
    cmd += ['--tc-mean-position', '15']
    cmd += ['--signal-duration', '16']
    cmd += ['--psd', 'psd-0.hdf']
    cmd += ['--signal-output', args.output]
    cmd += ['--not-generate-noise']
    if args.force:
        cmd += ['--force']
    if args.verbose:
        cmd += ['--verbose']
    run(cmd)
    return


if __name__ == "__main__":
    main()
