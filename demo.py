import pickle

from config import *
from data_gen import LoadDataset
from models import Seq2Seq


class adict(dict):
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self


if __name__ == '__main__':
    output_dir = data_path
    with open(os.path.join(output_dir, "mapping.pkl"), "wb") as fp:
        encode_table = pickle.load(fp)

    print(encode_table)

    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    encoder = checkpoint['encoder']
    decoder = checkpoint['decoder']
    encoder.eval()
    decoder.eval()
    model = Seq2Seq(encoder, decoder)

    test_loader = LoadDataset('test', text_only=False, data_path=data_path, batch_size=batch_size,
                              max_timestep=max_timestep, max_label_len=max_label_len, use_gpu=use_gpu, n_jobs=n_jobs,
                              train_set=train_set, dev_set=dev_set, test_set=test_set, dev_batch_size=dev_batch_size,
                              decode_beam_size=decode_beam_size)

    print(len(test_loader))

    for i, (x, y) in enumerate(test_loader):
        print(x.shape)
        print(y.shape)
        print(y)
        break
