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

## Results

|Model|CER|Download|
|---|---|---|
|Listen Attend Spell|16.2|[Link](https://github.com/foamliu/Speech-Transformer/releases/download/v1.0/BEST_checkpoint.tar)|


### Demo
```bash
$ python demo.py
```

|Audio|Out|GT|
|---|---|---|
|[audio_0.wav](https://github.com/foamliu/Listen-Attend-Spell/raw/master/audios/audio_0.wav)|同比前年增长五成|同比前年增长五成|
|[audio_1.wav](https://github.com/foamliu/Listen-Attend-Spell/raw/master/audios/audio_1.wav)|那么苹果能突破一万一大关吗|那么苹果能突破一万亿大关吗|
|[audio_2.wav](https://github.com/foamliu/Listen-Attend-Spell/raw/master/audios/audio_2.wav)|但在鼓励企业走出去方面是出多个信号|但在鼓励企业走出去方面释出多个信号|
|[audio_3.wav](https://github.com/foamliu/Listen-Attend-Spell/raw/master/audios/audio_3.wav)|他将尽全力带领被取得好成绩|他将尽全力带领队伍取得好成绩|
|[audio_4.wav](https://github.com/foamliu/Listen-Attend-Spell/raw/master/audios/audio_4.wav)|其中一些人是近年来国际拍卖会上艺术品的最大卖家|其中一些人是近年来国际拍卖会上艺术品的最大买家|
|[audio_5.wav](https://github.com/foamliu/Listen-Attend-Spell/raw/master/audios/audio_5.wav)|这种产品的设计仍需考虑文化历史和未来等因素|这种产品的设计仍需考虑文化历史和未来等因素|
|[audio_6.wav](https://github.com/foamliu/Listen-Attend-Spell/raw/master/audios/audio_6.wav)|东莞智战中国队首发再次变阵|中韩之战中国队首发再次变阵|
|[audio_7.wav](https://github.com/foamliu/Listen-Attend-Spell/raw/master/audios/audio_7.wav)|难怪他能接受年过半百的乔本的锁文|难怪他能接受年过半百的桥本的索吻|
|[audio_8.wav](https://github.com/foamliu/Listen-Attend-Spell/raw/master/audios/audio_8.wav)|就美国队中国产龙开发起双方调查|就美国对中国产轮胎发起双坊调查|
|[audio_9.wav](https://github.com/foamliu/Listen-Attend-Spell/raw/master/audios/audio_9.wav)|涉嫌编造谣言非法拘禁被刑拘|涉嫌编造谣言非法拘禁被刑拘|

## Reference
[1] W. Chan, N. Jaitly, Q. Le, and O. Vinyals, “Listen, attend and spell: A neural network for large vocabulary conversational speech recognition,” in ICASSP 2016. (https://arxiv.org/abs/1508.01211v2)

