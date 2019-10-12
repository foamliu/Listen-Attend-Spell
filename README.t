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
Aishell is an open-source Chinese Mandarin speech corpus published by Beijing Shell Shell Technology Co.,Ltd.

400 people from different accent areas in China are invited to participate in the recording, which is conducted in a quiet indoor environment using high fidelity microphone and downsampled to 16kHz. The manual transcription accuracy is above 95%, through professional speech annotation and strict quality inspection. The data is free for academic use. We hope to provide moderate amount of data for new researchers in the field of speech recognition.

```
@inproceedings{aishell_2017,
  title={AIShell-1: An Open-Source Mandarin Speech Corpus and A Speech Recognition Baseline},
  author={Hui Bu, Jiayu Du, Xingyu Na, Bengu Wu, Hao Zheng},
  booktitle={Oriental COCOSDA 2017},
  pages={Submitted},
  year={2017}
}
```

### Obtain
Create a data folder then run:
```bash
$ wget http://www.openslr.org/resources/33/data_aishell.tgz
```

## Dependencies
- Python 3.6.8
- PyTorch 1.3.0

## Usage

### Data wrangling
Extract data_aishell.tgz:
```bash
$ python extract.py
```

Extract wav files into train/dev/test folders:
```bash
$ cd data/data_aishell/wav
$ find . -name '*.tar.gz' -execdir tar -xzvf '{}' \;
```

Scan transcript data, generate features:
```bash
$ python pre_process.py
```

Now the folder structure under data folder is sth. like:

<pre>
data/
    data_aishell.tgz
    data_aishell/
        transcript/
            aishell_transcript_v0.8.txt
        wav/
            train/
            dev/
            test/
    aishell.pickle
</pre>

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

|Audio|Out|GT|
|---|---|---|
|[audio_0.wav](https://github.com/foamliu/Listen-Attend-Spell/raw/master/audios/audio_0.wav)|$(out_list_0)|$(gt_0)|
|[audio_1.wav](https://github.com/foamliu/Listen-Attend-Spell/raw/master/audios/audio_1.wav)|$(out_list_1)|$(gt_1)|
|[audio_2.wav](https://github.com/foamliu/Listen-Attend-Spell/raw/master/audios/audio_2.wav)|$(out_list_2)|$(gt_2)|
|[audio_3.wav](https://github.com/foamliu/Listen-Attend-Spell/raw/master/audios/audio_3.wav)|$(out_list_3)|$(gt_3)|
|[audio_4.wav](https://github.com/foamliu/Listen-Attend-Spell/raw/master/audios/audio_4.wav)|$(out_list_4)|$(gt_4)|
|[audio_5.wav](https://github.com/foamliu/Listen-Attend-Spell/raw/master/audios/audio_5.wav)|$(out_list_5)|$(gt_5)|
|[audio_6.wav](https://github.com/foamliu/Listen-Attend-Spell/raw/master/audios/audio_6.wav)|$(out_list_6)|$(gt_6)|
|[audio_7.wav](https://github.com/foamliu/Listen-Attend-Spell/raw/master/audios/audio_7.wav)|$(out_list_7)|$(gt_7)|
|[audio_8.wav](https://github.com/foamliu/Listen-Attend-Spell/raw/master/audios/audio_8.wav)|$(out_list_8)|$(gt_8)|
|[audio_9.wav](https://github.com/foamliu/Listen-Attend-Spell/raw/master/audios/audio_9.wav)|$(out_list_9)|$(gt_9)|

## Reference
[1] W. Chan, N. Jaitly, Q. Le, and O. Vinyals, “Listen, attend and spell: A neural network for large vocabulary conversational speech recognition,” in ICASSP 2016. (https://arxiv.org/abs/1508.01211v2)

