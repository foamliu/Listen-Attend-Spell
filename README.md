# Listen Attend and Spell

![apm](https://img.shields.io/apm/l/vim-mode.svg)

PyTorch implementation of Listen Attend and Spell Automatic Speech Recognition (ASR).
[paper](https://arxiv.org/abs/1508.01211).
```
@article{chan2015las,
title={Listen, Attend and Spell},
author={William Chan, Navdeep Jaitly, Quoc V. Le, Oriol Vinyals},
journal={arXiv:1508.01211},
year={2015}
}
```
## DataSet

### Introduction
The dataset I used is the LibriSpeech dataset. It contains about 1000 hours of 16kHz read English speech. It is available here: http://www.openslr.org/12/

### Obtain
Create a folder "data" then run:
```bash
$ wget http://www.openslr.org/resources/12/train-clean-100.tar.gz
$ wget http://www.openslr.org/resources/12/train-clean-360.tar.gz
$ wget http://www.openslr.org/resources/12/train-other-500.tar.gz
$ wget http://www.openslr.org/resources/12/dev-clean.tar.gz
$ wget http://www.openslr.org/resources/12/dev-other.tar.gz
$ wget http://www.openslr.org/resources/12/test-clean.tar.gz
$ wget http://www.openslr.org/resources/12/test-other.tar.gz
```

## Dependencies
- Python 3.6
- PyTorch 1.0.0

## Usage

### Data wrangling
Extract audio and transcript data, scan them, to get features:
```bash
$ python3 extract.py
$ python3 pre_process.py --data_path data/LibriSpeech/ --feature_type fbank --feature_dim 40 --output_path data/output --target char
```

### Train
```bash
$ python train.py
```

To visualize the training process：
```bash
$ tensorboard --logdir=runs
```

### Demo
```bash
$ python demo.py
```

## Results

## Reference