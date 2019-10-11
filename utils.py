import argparse
import logging

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


def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, loss, is_best):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'loss': loss,
             'model': model,
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


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def accuracy(scores, targets, k=1):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def parse_args():
    parser = argparse.ArgumentParser(description='Listen Attend and Spell')
    # Low Frame Rate (stacking and skipping frames)
    parser.add_argument('--LFR_m', default=4, type=int,
                        help='Low Frame Rate: number of frames to stack')
    parser.add_argument('--LFR_n', default=3, type=int,
                        help='Low Frame Rate: number of frames to skip')
    # general
    # Network architecture
    # encoder
    # TODO: automatically infer input dim
    parser.add_argument('--einput', default=80, type=int,
                        help='Dim of encoder input')
    parser.add_argument('--ehidden', default=256, type=int,
                        help='Size of encoder hidden units')
    parser.add_argument('--elayer', default=3, type=int,
                        help='Number of encoder layers.')
    parser.add_argument('--edropout', default=0.2, type=float,
                        help='Encoder dropout rate')
    parser.add_argument('--ebidirectional', default=True, type=bool,
                        help='Whether use bidirectional encoder')
    parser.add_argument('--etype', default='lstm', type=str,
                        help='Type of encoder RNN')
    # attention
    parser.add_argument('--atype', default='dot', type=str,
                        help='Type of attention (Only support Dot Product now)')
    # decoder
    parser.add_argument('--dembed', default=512, type=int,
                        help='Size of decoder embedding')
    parser.add_argument('--dhidden', default=512, type=int,
                        help='Size of decoder hidden units. Should be encoder '
                             '(2*) hidden size dependding on bidirection')
    parser.add_argument('--dlayer', default=1, type=int,
                        help='Number of decoder layers.')

    # Training config
    parser.add_argument('--epochs', default=20, type=int,
                        help='Number of maximum epochs')
    parser.add_argument('--half_lr', dest='half_lr', default=True, type=bool,
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
    parser.add_argument('--l2', default=1e-5, type=float,
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
    logger.setLevel(logging.INFO)
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


# [-0.5, 0.5]
def normalize(yt):
    yt_max = np.max(yt)
    yt_min = np.min(yt)
    a = 1.0 / (yt_max - yt_min)
    b = -(yt_max + yt_min) / (2 * (yt_max - yt_min))

    yt = yt * a + b
    return yt


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
def extract_feature(input_file, feature='fbank', dim=80, cmvn=True, delta=False, delta_delta=False,
                    window_size=25, stride=10, save_feature=None):
    y, sr = librosa.load(input_file, sr=None)
    yt, _ = librosa.effects.trim(y, top_db=20)
    yt = normalize(yt)
    ws = int(sr * 0.001 * window_size)
    st = int(sr * 0.001 * stride)
    if feature == 'fbank':  # log-scaled
        feat = librosa.feature.melspectrogram(y=yt, sr=sr, n_mels=dim,
                                              n_fft=ws, hop_length=st)
        feat = np.log(feat + 1e-6)
    elif feature == 'mfcc':
        feat = librosa.feature.mfcc(y=yt, sr=sr, n_mfcc=dim, n_mels=26,
                                    n_fft=ws, hop_length=st)
        feat[0] = librosa.feature.rmse(yt, hop_length=st, frame_length=ws)

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
