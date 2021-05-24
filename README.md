# Network intrusion detection system using deep learning

Deep Learning Models for `NIDS` using `NSL-KDD` and `ICIDS2017` datasets.

## Requirements

- `python 3.7`
- `pipenv` module

## Installation

Use `pipenv install` to install dependencies and `pipenv shell` to run the virtual environment.

## Models

Each model has its own file with this format `{model_name}.py`.

- `LSTM` model
- `GRU` model
- `RNN` model
- `DNN` model
- Classic machine learning models (Naive Bayes, Ada Boost and more)

## Usage

- Create a `data` directory at the root of the project if not exists.
- Put [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) dataset into `data/nsl` directory
- Put [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) dataset into `data/cicids/` directory
- Depending on your choices, these directories should be created into `data` directory: `mul-nsl`, `mul-cicids`, `bin-nsl` and `bin-cicids`.
- Run for each model using `python run.py`

#### Notes on running

- `CICIDS2017` is a pretty large dataset and processing it is time and memory consuming so for test purposes you can use the `NSL` dataset.