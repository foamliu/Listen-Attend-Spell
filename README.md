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
- Python 3.5.2
- PyTorch 1.0.0

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
|[audio_0.wav](https://github.com/foamliu/Listen-Attend-Spell/raw/master/audios/audio_0.wav)|影响同学<br>影响中学<br>永强中学<br>永强风讯<br>影响通讯|永强中学校长也是坠楼学生的语文老师|
|[audio_1.wav](https://github.com/foamliu/Listen-Attend-Spell/raw/master/audios/audio_1.wav)|事件近距离诊察和七号和七号和七号和七号和七年<br>事件近距离诊察和七号和七号和七号和七号和七号线<br>事件近距离正查和七号和七号和七号和七号和七号线<br>事件近距离诊察和七号和七号和七号和七年和前夫妻子<br>事件近距离正查和七号和七号和七号和七年和前夫妻子|实现近距离侦查和情报收集任务|
|[audio_2.wav](https://github.com/foamliu/Listen-Attend-Spell/raw/master/audios/audio_2.wav)|搜狐娱乐讯刘若瑄我们搜狐娱乐讯<br>搜狐娱乐讯刘若瑄为我们搜狐娱乐讯<br>搜狐娱乐讯明晚晚五万元<br>搜狐娱乐讯明我我们我们搜狐娱乐讯<br>搜狐娱乐讯明晚晚六万元|搜狐娱乐讯名为娱乐圈八卦的自媒体|
|[audio_3.wav](https://github.com/foamliu/Listen-Attend-Spell/raw/master/audios/audio_3.wav)|从股下来大改要七七七七七十七号线<br>从股权来打改要七七七七七十七号线<br>从股权来到改要七七七七七十七号线<br>从不下来打败了七七七七十七胜一负<br>从不下来打败了七七七七七十七胜一负|粗估下来大概要七十亿|
|[audio_4.wav](https://github.com/foamliu/Listen-Attend-Spell/raw/master/audios/audio_4.wav)|类似的补贴政策创造策超测超过七月份<br>来自地补贴政策创造策超测超过七月份<br>来自地补贴政策超测炒车场车场车场<br>类似的补贴政策差测超车场策长期认为<br>来自地补贴政策超测炒车场策略差距|类似的补贴政策常常是短效刺激|
|[audio_5.wav](https://github.com/foamliu/Listen-Attend-Spell/raw/master/audios/audio_5.wav)|该法发的大女三郎一地地<br>改跑八的大女三郎一地地<br>该话报道大女三郎一地地<br>该法发到大女三郎一地地<br>该法办的大女三郎一地地|白花花的大米洒了一地|
|[audio_6.wav](https://github.com/foamliu/Listen-Attend-Spell/raw/master/audios/audio_6.wav)|他们的财产和生活部股股股股股股股股股股股股股股股股股东的<br>他们的财产和生活部股股股股股股股股股股股股股股股票<br>他们的财产和生活部股股股股股股股股股股股股股股股股票<br>他们的财产和生活部股股股股股股股股股股股股股股股股股票<br>他们的财产和生活不股股股股股股股股股股股股股股股股票|他们的财产和生活不会受到太大影响|
|[audio_7.wav](https://github.com/foamliu/Listen-Attend-Spell/raw/master/audios/audio_7.wav)|大力发展现代<br>那里发展现代<br>大力发展阶段<br>到底发展现代<br>大利发展现代|大力发展现代农作物种业|
|[audio_8.wav](https://github.com/foamliu/Listen-Attend-Spell/raw/master/audios/audio_8.wav)|不不意味着港创发达股价的谈判的谈判<br>不不意味着港创发达股价的谈判的谈判的谈判<br>不不意味着港创发达股价的谈判的判决的谈判<br>不不意味着港创发达股价的谈判的谈判的谈判的谈判<br>不不意味着港创发达股价的谈判的谈判的判决的谈判|我们迎来了赶超发达国家的难得机遇|
|[audio_9.wav](https://github.com/foamliu/Listen-Attend-Spell/raw/master/audios/audio_9.wav)|推动七大地区价格下跌百分之价格<br>推动七大地区价格下跌百分之百分之价格<br>推动七大地区价格还会加快推动推动推动推动推动推动推动<br>推动七大地区价格还会加快推动推动推动推动推动推动推动推动<br>推动七大地区价格还会加快推动推动推动推动推动推动推动推动推动|推动其他地区加快发展|

## Reference
[1] W. Chan, N. Jaitly, Q. Le, and O. Vinyals, “Listen, attend and spell: A neural network for large vocabulary conversational speech recognition,” in ICASSP 2016. (https://arxiv.org/abs/1508.01211v2)

