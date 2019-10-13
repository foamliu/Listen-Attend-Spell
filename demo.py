import argparse
import pickle
import random
from shutil import copyfile

import torch

from config import device, pickle_file, input_dim
from data_gen import build_LFR_features
from utils import extract_feature, ensure_folder


def parse_args():
    parser = argparse.ArgumentParser(
        "End-to-End Automatic Speech Recognition Decoding.")
    # Low Frame Rate (stacking and skipping frames)
    parser.add_argument('--LFR_m', default=4, type=int,
                        help='Low Frame Rate: number of frames to stack')
    parser.add_argument('--LFR_n', default=3, type=int,
                        help='Low Frame Rate: number of frames to skip')
    # decode
    parser.add_argument('--beam_size', default=5, type=int,
                        help='Beam size')
    parser.add_argument('--nbest', default=1, type=int,
                        help='Nbest size')
    parser.add_argument('--decode_max_len', default=100, type=int,
                        help='Max output length. If ==0 (default), it uses a '
                             'end-detect function to automatically find maximum '
                             'hypothesis lengths')
    args = parser.parse_args()
    return args


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
    ensure_folder('audios')

    results = []

    for i, sample in enumerate(samples):
        wave = sample['wave']
        trn = sample['trn']

        copyfile(wave, 'audios/audio_{}.wav'.format(i))

        input = extract_feature(input_file=wave, feature='fbank', dim=input_dim, cmvn=True)
        input = build_LFR_features(input, m=args.LFR_m, n=args.LFR_n)
        # print(input.shape)

        # input = np.expand_dims(input, axis=0)
        input = torch.from_numpy(input).to(device)
        input_length = [input.shape[0]]
        input_length = torch.LongTensor(input_length).to(device)

        with torch.no_grad():
            nbest_hyps = model.recognize(input, input_length, char_list, args)

        out_list = []
        for hyp in nbest_hyps:
            out = hyp['yseq']
            out = [char_list[idx] for idx in out]
            out = ''.join(out).replace('<sos>', '').replace('<eos>', '')
            out_list.append(out)
        out = out_list[0]
        print('OUT: {}'.format(out))

        gt = [char_list[idx] for idx in trn]
        gt = ''.join(gt).replace('<eos>', '')
        print(' GT: {}\n'.format(gt))

        results.append({'out_list_{}'.format(i): out_list, 'gt_{}'.format(i): gt})

    import json

    with open('results.json', 'w') as file:
        json.dump(results, file, indent=4, ensure_ascii=False)
