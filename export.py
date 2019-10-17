import torch

from models.seq2seq import Seq2Seq

if __name__ == '__main__':
    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    # model.eval()

    filename = 'listen-attend-spell.pt'
    torch.save(model.state_dict(), filename)

    model = Seq2Seq()
    model.load_state_dict(torch.load(filename))
