import argparse
import logging
import os
import pickle
from operator import itemgetter

import editdistance as ed
import librosa
import numpy as np
import torch


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(epoch, epochs_since_improvement, encoder, decoder, optimizer, loss, is_best):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'loss': loss,
             'encoder': encoder,
             'decoder': decoder,
             'optimizer': optimizer}

    filename = 'checkpoint.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_checkpoint.tar')


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k=1):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def parse_args():
    parser = argparse.ArgumentParser(description='Listen Attend and Spell')
    # general
    # Network architecture
    # encoder
    # TODO: automatically infer input dim
    parser.add_argument('--einput', default=80, type=int,
                        help='Dim of encoder input')
    parser.add_argument('--ehidden', default=512, type=int,
                        help='Size of encoder hidden units')
    parser.add_argument('--elayer', default=4, type=int,
                        help='Number of encoder layers.')
    parser.add_argument('--edropout', default=0.0, type=float,
                        help='Encoder dropout rate')
    parser.add_argument('--ebidirectional', default=1, type=int,
                        help='Whether use bidirectional encoder')
    parser.add_argument('--etype', default='lstm', type=str,
                        help='Type of encoder RNN')
    # attention
    parser.add_argument('--atype', default='dot', type=str,
                        help='Type of attention (Only support Dot Product now)')
    # decoder
    parser.add_argument('--dembed', default=512, type=int,
                        help='Size of decoder embedding')
    parser.add_argument('--dhidden', default=512 * 2, type=int,
                        help='Size of decoder hidden units. Should be encoder '
                             '(2*) hidden size dependding on bidirection')
    parser.add_argument('--dlayer', default=1, type=int,
                        help='Number of decoder layers.')

    # Training config
    parser.add_argument('--epochs', default=30, type=int,
                        help='Number of maximum epochs')
    parser.add_argument('--half_lr', dest='half_lr', default=0, type=int,
                        help='Halving learning rate when get small improvement')
    parser.add_argument('--early_stop', dest='early_stop', default=0, type=int,
                        help='Early stop training when halving lr but still get'
                             'small improvement')
    parser.add_argument('--max_norm', default=5, type=float,
                        help='Gradient norm threshold to clip')
    # minibatch
    parser.add_argument('--batch-size', '-b', default=32, type=int,
                        help='Batch size')
    parser.add_argument('--maxlen_in', default=800, type=int, metavar='ML',
                        help='Batch size is reduced if the input sequence length > ML')
    parser.add_argument('--maxlen_out', default=150, type=int, metavar='ML',
                        help='Batch size is reduced if the output sequence length > ML')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to generate minibatch')
    # optimizer
    parser.add_argument('--optimizer', default='adam', type=str,
                        choices=['sgd', 'adam'],
                        help='Optimizer (support sgd and adam now)')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Init learning rate')
    parser.add_argument('--momentum', default=0.0, type=float,
                        help='Momentum for optimizer')
    parser.add_argument('--l2', default=0.0, type=float,
                        help='weight decay (L2 penalty)')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')

    args = parser.parse_args()
    return args


def get_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s \t%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


def ensure_folder(folder):
    import os
    if not os.path.isdir(folder):
        os.mkdir(folder)


def pad_list(xs, pad_value):
    # From: espnet/src/nets/e2e_asr_th.py: pad_list()
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad


# Acoustic Feature Extraction
# Parameters
#     - input file  : str, audio file path
#     - feature     : str, fbank or mfcc
#     - dim         : int, dimension of feature
#     - cmvn        : bool, apply CMVN on feature
#     - window_size : int, window size for FFT (ms)
#     - stride      : int, window stride for FFT
#     - save_feature: str, if given, store feature to the path and return len(feature)
# Return
#     acoustic features with shape (time step, dim)
def extract_feature(input_file, feature='fbank', dim=40, cmvn=True, delta=False, delta_delta=False,
                    window_size=25, stride=10, save_feature=None):
    y, sr = librosa.load(input_file, sr=None)
    ws = int(sr * 0.001 * window_size)
    st = int(sr * 0.001 * stride)
    if feature == 'fbank':  # log-scaled
        feat = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=dim,
                                              n_fft=ws, hop_length=st)
        feat = np.log(feat + 1e-6)
    elif feature == 'mfcc':
        feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=dim, n_mels=26,
                                    n_fft=ws, hop_length=st)
        feat[0] = librosa.feature.rmse(y, hop_length=st, frame_length=ws)

    else:
        raise ValueError('Unsupported Acoustic Feature: ' + feature)

    feat = [feat]
    if delta:
        feat.append(librosa.feature.delta(feat[0]))

    if delta_delta:
        feat.append(librosa.feature.delta(feat[0], order=2))
    feat = np.concatenate(feat, axis=0)
    if cmvn:
        feat = (feat - feat.mean(axis=1)[:, np.newaxis]) / (feat.std(axis=1) + 1e-16)[:, np.newaxis]
    if save_feature is not None:
        tmp = np.swapaxes(feat, 0, 1).astype('float32')
        np.save(save_feature, tmp)
        return len(tmp)
    else:
        return np.swapaxes(feat, 0, 1).astype('float32')


# Target Encoding Function
# Parameters
#     - input list : list, list of target list
#     - table      : dict, token-index table for encoding (generate one if it's None)
#     - mode       : int, encoding mode ( phoneme / char / subword / word )
#     - max idx    : int, max encoding index (0=<sos>, 1=<eos>, 2=<unk>)
# Return
#     - output list: list, list of encoded targets
#     - output dic : dict, token-index table used during encoding
def encode_target(input_list, table=None, mode='subword', max_idx=500):
    if table is None:
        ### Step 1. Calculate wrd frequency
        table = {}
        for target in input_list:
            for t in target:
                if t not in table:
                    table[t] = 1
                else:
                    table[t] += 1
        ### Step 2. Top k list for encode map
        max_idx = min(max_idx - 3, len(table))
        all_tokens = [k for k, v in sorted(table.items(), key=itemgetter(1), reverse=True)][:max_idx]
        table = {'<sos>': 0, '<eos>': 1}
        if mode == "word": table['<unk>'] = 2
        for tok in all_tokens:
            table[tok] = len(table)
    ### Step 3. Encode
    output_list = []
    for target in input_list:
        tmp = [0]
        for t in target:
            if t in table:
                tmp.append(table[t])
            else:
                if mode == "word":
                    tmp.append(2)
                else:
                    tmp.append(table['<unk>'])
                    # raise ValueError('OOV error: '+t)
        tmp.append(1)
        output_list.append(tmp)
    return output_list, table


# Target Padding Function
# Parameters
#     - y          : list, list of int
#     - max_len    : int, max length of output (0 for max_len in y)
# Return
#     - new_y      : np.array with shape (len(y),max_len)
def target_padding(y, max_len):
    if max_len is 0: max_len = max([len(v) for v in y])
    new_y = np.zeros((len(y), max_len), dtype=int)
    for idx, label_seq in enumerate(y):
        new_y[idx, :len(label_seq)] = np.array(label_seq)
    return new_y


class Mapper():
    '''Mapper for index2token'''

    def __init__(self, file_path):
        # Find mapping
        with open(os.path.join(file_path, 'mapping.pkl'), 'rb') as fp:
            self.mapping = pickle.load(fp)
        self.r_mapping = {v: k for k, v in self.mapping.items()}
        symbols = ''.join(list(self.mapping.keys()))
        if '▁' in symbols:
            self.unit = 'subword'
        elif '#' in symbols:
            self.unit = 'phone'
        elif len(self.mapping) < 50:
            self.unit = 'char'
        else:
            self.unit = 'word'

    def get_dim(self):
        return len(self.mapping)

    def translate(self, seq, return_string=False):
        new_seq = []
        for c in trim_eos(seq):
            new_seq.append(self.r_mapping[c])

        if return_string:
            if self.unit == 'subword':
                new_seq = ''.join(new_seq).replace('<sos>', '').replace('<eos>', '').replace('▁', ' ').lstrip()
            elif self.unit == 'word':
                new_seq = ' '.join(new_seq).replace('<sos>', '').replace('<eos>', '').lstrip()
            elif self.unit == 'phone':
                new_seq = ' '.join(collapse_phn(new_seq)).replace('<sos>', '').replace('<eos>', '')
            elif self.unit == 'char':
                new_seq = ''.join(new_seq).replace('<sos>', '').replace('<eos>', '')
        return new_seq


def cal_acc(pred, label):
    pred = np.argmax(pred.cpu().detach(), axis=-1)
    label = label.cpu()
    accs = []
    for p, l in zip(pred, label):
        correct = 0.0
        total_char = 0
        for pp, ll in zip(p, l):
            if ll == 0: break
            correct += int(pp == ll)
            total_char += 1
        accs.append(correct / total_char)
    return sum(accs) / len(accs)


def cal_cer(pred, label, mapper, get_sentence=False, argmax=True):
    if argmax:
        pred = np.argmax(pred.cpu().detach(), axis=-1)
    label = label.cpu()
    pred = [mapper.translate(p, return_string=True) for p in pred]
    label = [mapper.translate(l, return_string=True) for l in label]

    if get_sentence:
        return pred, label
    eds = [float(ed.eval(p.split(' '), l.split(' '))) / len(l.split(' ')) for p, l in zip(pred, label)]

    return sum(eds) / len(eds)


# Only draw first attention head
def draw_att(att_list, hyp_txt):
    attmaps = []
    for att, hyp in zip(att_list[0], np.argmax(hyp_txt.cpu().detach(), axis=-1)):
        att_len = len(trim_eos(hyp))
        att = att.detach().cpu()
        attmaps.append(torch.stack([att, att, att], dim=0)[:, :att_len, :])  # +1 for att. @ <eos>
    return attmaps


def collapse_phn(seq):
    # phonemes = ["b", "bcl", "d", "dcl", "g", "gcl", "p", "pcl", "t", "tcl", "k", "kcl", "dx", "q", "jh", "ch", "s", "sh", "z", "zh",
    # "f", "th", "v", "dh", "m", "n", "ng", "em", "en", "eng", "nx", "l", "r", "w", "y",
    # "hh", "hv", "el", "iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay", "ah", "ao", "oy",
    # "ow", "uh", "uw", "ux", "er", "ax", "ix", "axr", "ax-h", "pau", "epi", "h#"]

    phonemse_reduce_mapping = {"b": "b", "bcl": "h#", "d": "d", "dcl": "h#", "g": "g", "gcl": "h#", "p": "p",
                               "pcl": "h#", "t": "t", "tcl": "h#", "k": "k", "kcl": "h#", "dx": "dx", "q": "q",
                               "jh": "jh", "ch": "ch", "s": "s", "sh": "sh", "z": "z", "zh": "sh",
                               "f": "f", "th": "th", "v": "v", "dh": "dh", "m": "m", "n": "n", "ng": "ng", "em": "m",
                               "en": "n", "eng": "ng", "nx": "n", "l": "l", "r": "r", "w": "w", "y": "y",
                               "hh": "hh", "hv": "hh", "el": "l", "iy": "iy", "ih": "ih", "eh": "eh", "ey": "ey",
                               "ae": "ae", "aa": "aa", "aw": "aw", "ay": "ay", "ah": "ah", "ao": "aa", "oy": "oy",
                               "ow": "ow", "uh": "uh", "uw": "uw", "ux": "uw", "er": "er", "ax": "ah", "ix": "ih",
                               "axr": "er", "ax-h": "ah", "pau": "h#", "epi": "h#", "h#": "h#", "<sos>": "<sos>",
                               "<unk>": "<unk>", "<eos>": "<eos>"}

    return [phonemse_reduce_mapping[c] for c in seq]


def trim_eos(seqence):
    new_pred = []
    for char in seqence:
        new_pred.append(int(char))
        if char == 1:
            break
    return new_pred
