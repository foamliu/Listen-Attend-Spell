import os

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

# Model parameters
input_dim = 40  # dimension of feature
window_size = 25  # window size for FFT (ms)
hidden_size = 512
embedding_dim = 512
stride = 10  # window stride for FFT
cmvn = True  # apply CMVN on feature
num_layers = 4

# Training parameters
dataset = 'librispeech'
data_path = 'data/output/libri_fbank80_char30'
n_jobs = 8
max_timestep = 3000
max_label_len = 400
train_set = ['train-clean-100']
dev_set = ['dev-clean']
test_set = ['test-clean']
batch_size = 32
dev_batch_size = 16
use_gpu = True
decode_beam_size = 20
tf_rate = 1.0

lr = 1e-3
num_workers = 1  # for data-loading; right now, only 1 works with h5py
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 10  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none

# Data parameters
PAD_token = 0
sos_id = 1
eos_id = 2
IGNORE_ID = -1
num_train = 120418
num_dev = 14326
num_test = 7176
vocab_size = 30
HALF_BATCHSIZE_TIME = 800
HALF_BATCHSIZE_LABEL = 150

DATA_DIR = 'data'
aishell_folder = 'data/data_aishell'
wav_folder = os.path.join(aishell_folder, 'wav')
tran_file = os.path.join(aishell_folder, 'transcript/aishell_transcript_v0.8.txt')
IMG_DIR = 'data/images'
pickle_file = 'data/aishell.pickle'
