import pickle
import os
import random
import numpy as np
import torch
from config import pickle_file
from utils import extract_feature, parse_args


class adict(dict):
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self


if __name__ == '__main__':
    args = parse_args()
    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)
    char_list = data['IVOCAB']
    samples = data['test']

    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model.eval()

    samples = random.sample(samples, 10)

    for sample in samples:
        wave = sample['wave']
        trn = sample['trn']

        input = extract_feature(input_file=wave, feature='fbank', dim=512)
        input = np.expand_dims(input, axis=0)
        input_length = [input[0].shape[0]]
        nbest_hyps = model.recognize(input, input_length, char_list, args)

        print(nbest_hyps)
