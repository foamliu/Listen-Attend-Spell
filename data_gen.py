import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from config import num_workers, pickle_file, IGNORE_ID
from utils import extract_feature, parse_args


def pad_collate(batch):
    max_input_len = float('-inf')
    max_target_len = float('-inf')

    for elem in batch:
        feature, trn = elem
        max_input_len = max_input_len if max_input_len > feature.shape[0] else feature.shape[0]
        max_target_len = max_target_len if max_target_len > len(trn) else len(trn)

    for i, elem in enumerate(batch):
        feature, trn = elem
        input_length = feature.shape[0]
        input_dim = feature.shape[1]
        padded_input = np.zeros((max_input_len, input_dim), dtype=np.float32)
        padded_input[:input_length, :] = feature
        padded_target = np.pad(trn, (0, max_target_len - len(trn)), 'constant', constant_values=IGNORE_ID)
        batch[i] = (padded_input, padded_target, input_length)

    # sort it by input lengths (long to short)
    batch.sort(key=lambda x: x[2], reverse=True)

    return default_collate(batch)


class AiShellDataset(Dataset):
    def __init__(self, args, split):
        with open(pickle_file, 'rb') as file:
            data = pickle.load(file)

        self.samples = data[split]
        self.args = args
        print('loading {} {} samples...'.format(len(self.samples), split))

    def __getitem__(self, i):
        sample = self.samples[i]
        wave = sample['wave']
        trn = sample['trn']

        feature = extract_feature(input_file=wave, feature='fbank', dim=self.args.einput)

        return feature, trn

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    args = parse_args()
    train_dataset = AiShellDataset(args, 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=num_workers,
                                               pin_memory=True, collate_fn=pad_collate)

    print('len(train_dataset): ' + str(len(train_dataset)))
    print('len(train_loader):' + str(len(train_loader)))

    feature = train_dataset[10][0]
    print('feature.shape: ' + str(feature.shape))

    trn = train_dataset[10][1]
    print('trn: ' + str(trn))

    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)
    char_list = data['IVOCAB']

    trn = [char_list[idx] for idx in trn]
    trn = ''.join(trn)
    print(trn)

    for data in train_loader:
        padded_input, padded_target, input_lengths = data
        print('padded_input: ' + str(padded_input))
        print('padded_target: ' + str(padded_target))
        print('input_lengths: ' + str(input_lengths))
        break
