import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from config import device, grad_clip, print_freq, vocab_size, num_workers, sos_id, eos_id
from data_gen import AiShellDataset, pad_collate
from models.decoder import Decoder
from models.encoder import Encoder
from models.seq2seq import Seq2Seq
from utils import parse_args, save_checkpoint, AverageMeter, clip_gradient, get_logger, adjust_learning_rate, get_learning_rate


def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_loss = float('inf')
    writer = SummaryWriter()
    epochs_since_improvement = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        # model
        encoder = Encoder(args.einput, args.ehidden, args.elayer,
                          dropout=args.edropout, bidirectional=args.ebidirectional,
                          rnn_type=args.etype)
        decoder = Decoder(vocab_size, args.dembed, sos_id,
                          eos_id, args.dhidden, args.dlayer,
                          bidirectional_encoder=args.ebidirectional)
        model = Seq2Seq(encoder, decoder)
        print(model)
        model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    logger = get_logger()

    # Custom dataloaders
    train_dataset = AiShellDataset(args, 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=pad_collate,
                                               pin_memory=True, shuffle=True, num_workers=num_workers)
    valid_dataset = AiShellDataset(args, 'dev')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=pad_collate,
                                               pin_memory=True, shuffle=False, num_workers=num_workers)

    # Epochs
    for epoch in range(start_epoch, args.epochs):
        # Halving learning rate when get small improvement
        if args.half_lr and epochs_since_improvement > 0:
            adjust_learning_rate(optimizer, 0.5)

        # One epoch's training
        train_loss = train(train_loader=train_loader,
                           model=model,
                           optimizer=optimizer,
                           epoch=epoch,
                           logger=logger)
        writer.add_scalar('Train_Loss', train_loss, epoch)

        lr = get_learning_rate(optimizer)
        print('Learning rate: {}\n'.format(lr))
        writer.add_scalar('Learning_Rate', lr, epoch)

        # One epoch's validation
        valid_loss = valid(valid_loader=valid_loader,
                           model=model,
                           logger=logger)
        writer.add_scalar('Valid_Loss', valid_loss, epoch)

        # Check if there was an improvement
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best)


def train(train_loader, model, optimizer, epoch, logger):
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()

    # Batches
    for i, (data) in enumerate(train_loader):
        # Move to GPU, if available
        padded_input, padded_target, input_lengths = data
        padded_input = padded_input.to(device)
        padded_target = padded_target.to(device)
        input_lengths = input_lengths.to(device)

        # Forward prop.
        loss = model(padded_input, input_lengths, padded_target)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        clip_gradient(optimizer, grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())

        # Print status
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader), loss=losses))

    return losses.avg


def valid(valid_loader, model, logger):
    model.eval()

    losses = AverageMeter()

    # Batches
    for data in tqdm(valid_loader):
        # Move to GPU, if available
        padded_input, padded_target, input_lengths = data
        padded_input = padded_input.to(device)
        padded_target = padded_target.to(device)
        input_lengths = input_lengths.to(device)

        # Forward prop.
        loss = model(padded_input, input_lengths, padded_target)

        # Keep track of metrics
        losses.update(loss.item())

    # Print status
    logger.info('\nValidation Loss {loss.val:.4f} ({loss.avg:.4f})\n'.format(loss=losses))

    return losses.avg


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
