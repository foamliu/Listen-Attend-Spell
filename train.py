import numpy as np
from torch import nn

from config import *
from data_gen import LoadDataset
from models import Encoder, Decoder, Seq2Seq
from utils import parse_args, save_checkpoint, AverageMeter, clip_gradient, get_logger, Mapper, cal_acc, cal_cer

VAL_STEP = 30  # Additional Inference Timesteps to run during validation (to calculate CER)


def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_loss = float('inf')
    epochs_since_improvement = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        encoder = Encoder(args.input_dim, args.encoder_hidden_size, args.num_layers)
        decoder = Decoder(vocab_size, args.embedding_dim, args.decoder_hidden_size)

        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)

        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD([{'params': encoder.parameters()}, {'params': decoder.parameters()}],
                                        lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam([{'params': encoder.parameters()}, {'params': decoder.parameters()}],
                                         lr=args.lr)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        encoder = checkpoint['encoder']
        decoder = checkpoint['decoder']
        optimizer = checkpoint['optimizer']

    logger = get_logger()

    # Move to GPU, if available
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Custom dataloaders
    train_loader = LoadDataset('train', text_only=False, data_path=data_path, batch_size=batch_size,
                               max_timestep=max_timestep, max_label_len=max_label_len, use_gpu=use_gpu, n_jobs=n_jobs,
                               train_set=train_set, dev_set=dev_set, test_set=test_set, dev_batch_size=dev_batch_size,
                               decode_beam_size=decode_beam_size)

    val_loader = LoadDataset('dev', text_only=False, data_path=data_path, batch_size=batch_size,
                             max_timestep=max_timestep, max_label_len=max_label_len, use_gpu=use_gpu, n_jobs=n_jobs,
                             train_set=train_set, dev_set=dev_set, test_set=test_set, dev_batch_size=dev_batch_size,
                             decode_beam_size=decode_beam_size)

    seq_loss = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none').to(device)  # , reduction='none')

    # Epochs
    for epoch in range(start_epoch, args.end_epoch):
        # One epoch's training
        train_loss = train(train_loader=train_loader,
                           encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer,
                           epoch=epoch,
                           logger=logger,
                           seq_loss=seq_loss)
        logger.info('[Training] Accuracy : {:.4f}'.format(train_loss))

        # One epoch's validation
        valid_loss = valid(valid_loader=val_loader,
                           encoder=encoder,
                           decoder=decoder,
                           seq_loss=seq_loss)

        logger.info('[Validate] Accuracy : {:.4f}'.format(valid_loss))

        # Check if there was an improvement
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, encoder, decoder, optimizer, best_loss, is_best)


def train(train_loader, encoder, decoder, optimizer, epoch, logger, seq_loss):
    encoder.train()  # train mode (dropout and batchnorm is used)
    decoder.train()

    model = Seq2Seq(encoder, decoder)

    losses = AverageMeter()

    # Batches
    for i, (x, y) in enumerate(train_loader):
        # Hack bucket, record state length for each uttr, get longest label seq for decode step
        assert len(x.shape) == 4, 'Bucketing should cause acoustic feature to have shape 1xBxTxD'
        assert len(y.shape) == 3, 'Bucketing should cause label have to shape 1xBxT'
        x = x.squeeze(0).to(device=device, dtype=torch.float32)
        y = y.squeeze(0).to(device=device, dtype=torch.long)
        state_len = np.sum(np.sum(x.cpu().data.numpy(), axis=-1) != 0, axis=-1)
        state_len = [int(sl) for sl in state_len]
        ans_len = int(torch.max(torch.sum(y != 0, dim=-1)))

        # ASR forwarding
        optimizer.zero_grad()
        ctc_pred, state_len, att_pred, _ = model(x, ans_len, y)

        # Calculate loss function
        label = y[:, 1:ans_len + 1].contiguous()

        # CE loss on attention decoder
        b, t, c = att_pred.shape
        att_loss = seq_loss(att_pred.view(b * t, c), label.view(-1))
        att_loss = torch.sum(att_loss.view(b, t), dim=-1) / torch.sum(y != 0, dim=-1) \
            .to(device=device, dtype=torch.float32)  # Sum each uttr and devide by length
        att_loss = torch.mean(att_loss)  # Mean by batch
        loss = att_loss

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


def valid(valid_loader, encoder, decoder, seq_loss):
    encoder.eval()
    decoder.eval()

    model = Seq2Seq(encoder, decoder)

    mapper = Mapper(data_path)

    losses = AverageMeter()

    # Init stats
    val_loss, val_ctc, val_att, val_acc, val_cer = 0.0, 0.0, 0.0, 0.0, 0.0
    val_len = 0
    all_pred, all_true = [], []

    with torch.no_grad():
        # Batches
        for i, (x, y) in enumerate(valid_loader):
            # Prepare data
            if len(x.shape) == 4: x = x.squeeze(0)
            if len(y.shape) == 3: y = y.squeeze(0)
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)
            state_len = torch.sum(torch.sum(x.cpu(), dim=-1) != 0, dim=-1)
            state_len = [int(sl) for sl in state_len]
            ans_len = int(torch.max(torch.sum(y != 0, dim=-1)))

            # Forward
            ctc_pred, state_len, att_pred, att_maps = model(x, ans_len + VAL_STEP, state_len=state_len)

            # Compute attention loss & get decoding results
            label = y[:, 1:ans_len + 1].contiguous()

            seq_loss = seq_loss(att_pred[:, :ans_len, :].contiguous().view(-1, att_pred.shape[-1]), label.view(-1))
            seq_loss = torch.sum(seq_loss.view(x.shape[0], -1), dim=-1) / torch.sum(y != 0, dim=-1) \
                .to(device=device, dtype=torch.float32)  # Sum each uttr and devide by length
            seq_loss = torch.mean(seq_loss)  # Mean by batch
            val_att += seq_loss.detach() * int(x.shape[0])
            t1, t2 = cal_cer(att_pred, label, mapper=mapper, get_sentence=True)
            all_pred += t1
            all_true += t2
            val_acc += cal_acc(att_pred, label) * int(x.shape[0])
            val_cer += cal_cer(att_pred, label, mapper=mapper) * int(x.shape[0])

            loss = val_att
            # Keep track of metrics
            losses.update(loss.item())

    return losses.avg


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
