# CNN-coinc
This repository contains the code used in the CNN-Coinc submission for the
machine learning gravitational-wave search mock data challenge
[(MLGWSC-1)](https://github.com/gwastro/ml-mock-data-challenge-1). It provides
both the code used for training and for evaluation.

The model used in the challenge is stored in the folder `model`.

## Requirements
This code was written and tested with `Python 3.7.3`.

To run the code, clone this repository
```
git clone https://github.com/MarlinSchaefer/cnn-coinc
```
and change into the repository
```
cd cnn_coinc
```
Afterwards, install the requirements (preferably in a virtual environment)
```
pip install -r requirements.txt
```

## Training
To train the network, training data is required. After that data has been
generated, the network can be trained with a simple call to the `train.py`
script.

### Data generation
Make sure to store the signal and noise files in the same directory.

!!! THE SIGNAL FILES HAVE TO BE CALLED `signals-{i}.hdf`, WHERE `i` IS REPLACED
BY THE INDEX OF THE NUMBER OF THE FILE !!!
To generate signals use the `gen_signals.py` script.

!!! THE OUTPUT NOISE FILE HAS TO BE CALLED `noise.py` !!!
To generate noise in the way used for the MLGWSC-1 submission, the real noise
file from the challenge has to be downloaded (~90GB). To do so, run the
`download_data.py` script.
After it has been downloaded, we need to whiten the noise. Care, the output file
will be large (~200GB). To whiten the data, use the `whiten_data.py` script.

### Running the training
Point the `train.py` script to the data-directory created in the previous step.


## Evaluation
To run the network on test-data generated by the `generate_data.py` script of
the [MLGWSC-1](https://github.com/gwastro/ml-mock-data-challenge-1/blob/master/generate_data.py),
run the `executable.py` script.


## Citation
If you make use of the code in this repository, please cite the MLGWSC-1
paper:
<insert-citation>
