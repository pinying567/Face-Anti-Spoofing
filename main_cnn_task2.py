import numpy as np
import argparse
import os
import json
import torch.utils.data as data
from torchvision import transforms
import random
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import pdb

from dataset_img_task2 import AntiSpoofingDataset
from model_task2 import CNN
from loss import MCFocalLoss
from util import averageMeter, lr_decay, accuracy

def main():
    global save_dir, logger

    # setup random seed
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)

    # setup directory to save logfiles, checkpoints, and output csv
    save_dir = args.save_dir
    if 'train' in args.phase and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # setup logger
    logger = None
    if 'train' in args.phase:
        logger = open(os.path.join(save_dir, 'train.log'), 'a')
        logfile = os.path.join(save_dir, 'training_log.json')
        log = {'train': []}
        logger.write('{}\n'.format(args))

    # setup data loader for training images
    if args.phase == 'train':
        dataset_train = AntiSpoofingDataset(os.path.join(args.data_root, 'train'))
        train_loader = data.DataLoader(dataset_train, shuffle=True, drop_last=False, pin_memory=True, batch_size=args.batch_size, num_workers=4)
        print('train: {}'.format(dataset_train.__len__()))
        logger.write('train: {}\n'.format(dataset_train.__len__()))

    # setup data loader for validation/testing images
    if args.test_dir:
        dataset_val = AntiSpoofingDataset(args.test_dir)
    else:
        dataset_val = AntiSpoofingDataset(os.path.join(args.data_root, 'val'))

    print('val/test: {}'.format(dataset_val.__len__()))
    if args.phase == 'train':
        logger.write('val/test: {}\n'.format(dataset_val.__len__()))

    val_loader = data.DataLoader(dataset_val, shuffle=False, drop_last=False, batch_size=args.batch_size, num_workers=4)

    # setup model
    model = CNN().cuda()

    # setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # setup criterion
    criterion = MCFocalLoss()

    # load checkpoint
    start_ep = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['opt_state'])
        start_ep = checkpoint['epoch']
        print("Loaded checkpoint '{}' (epoch: {})".format(args.checkpoint, start_ep))

        if args.phase == 'train':
            logger.write("Loaded checkpoint '{}' (epoch: {})\n".format(args.checkpoint, start_ep))
            if os.path.isfile(logfile):
                log = json.load(open(logfile, 'r'))

    if args.phase == 'train':
        # start training
        print('Start training from epoch {}'.format(start_ep))
        logger.write('Start training from epoch {}\n'.format(start_ep))

        for epoch in range(start_ep, args.epochs):

            acc, loss = train(train_loader, model, optimizer, epoch, criterion)
            log['train'].append([epoch + 1, acc, loss])

            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    acc_val, loss_val = val(val_loader, model, criterion)

                # save checkpoint
                state = {
                    'epoch': epoch + 1,
                    'acc': acc,
                    'model_state': model.state_dict(),
                    'opt_state': optimizer.state_dict()
                }
                checkpoint = os.path.join(save_dir, 'ep-{}.pkl'.format(epoch + 1))
                torch.save(state, checkpoint)
                print('[Checkpoint] {} is saved.'.format(checkpoint))
                logger.write('[Checkpoint] {} is saved.\n'.format(checkpoint))
                json.dump(log, open(logfile, 'w'))

            if (epoch + 1) % args.step == 0:
                lr_decay(optimizer, decay_rate=args.gamma)

        # save last model
        state = {
            'epoch': epoch + 1,
            'acc': acc,
            'model_state': model.state_dict(),
            'opt_state': optimizer.state_dict()
        }
        checkpoint = os.path.join(save_dir, 'last_checkpoint.pkl')
        torch.save(state, checkpoint)
        print('[Checkpoint] {} is saved.'.format(checkpoint))
        logger.write('[Checkpoint] {} is saved.\n'.format(checkpoint))
        print('Training is done.')
        logger.write('Training is done.\n')
        logger.close()

    else:
        with torch.no_grad():
            acc_val, loss_val = val(val_loader, model, criterion, save_result=True)

        print('Testing is done.')

def train(data_loader, model, optimizer, epoch, criterion):

    losses = averageMeter()
    ACC = averageMeter()

    # setup training mode
    model.train()

    for (step, value) in enumerate(data_loader):

        image = value[0].cuda()
        target = value[2].cuda(non_blocking=True)

        # forward
        output = model(image).squeeze()

        # compute loss
        loss = criterion(output, target)
        losses.update(loss.item(), image.size(0))

        # compute acc
        acc = accuracy(output, target, topk=(1,))[0]
        ACC.update(acc.item(), image.size(0))

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # logging
    curr_lr = optimizer.param_groups[0]['lr']
    print('Epoch: [{}/{}]\t' \
        'LR: [{:.6g}]\t' \
        'Loss {loss.avg:.4f}\t' \
        'Acc {acc.avg:.3f}'.format(
            epoch + 1, args.epochs, curr_lr, loss=losses, acc=ACC
        )
    )
    logger.write('Epoch: [{}/{}]\t' \
        'LR: [{:.6g}]\t' \
        'Loss {loss.avg:.4f}\t' \
        'Acc {acc.avg:.3f}\n'.format(
            epoch + 1, args.epochs, curr_lr, loss=losses, acc=ACC
        )
    )
    return ACC.avg, losses.avg


def val(data_loader, model, criterion, save_result=False):

    losses = averageMeter()
    ACC = averageMeter()

    all_labels = []
    all_preds = []
    if save_result:
        fnames = []

    # setup evaluation mode
    model.eval()

    for (step, value) in enumerate(data_loader):

        image = value[0].cuda()
        if len(value) > 2:
            target = value[2].cuda(non_blocking=True)
            all_labels = np.concatenate((all_labels, value[2]))

        # forward
        output = model(image).squeeze()

        # accumulate image_id & predictions
        if save_result:
            fnames.extend(value[1])

        pred = torch.max(output, dim=1)[1].data.cpu().numpy()
        all_preds = np.concatenate((all_preds, pred), axis=0)
        if len(value) > 2:
            # compute acc
            acc = accuracy(output, target, topk=(1,))[0]
            ACC.update(acc.item(), image.size(0))

            # compute loss
            loss = criterion(output, target)
            losses.update(loss.item(), image.size(0))

    # logging
    print('[Val] Loss {loss.avg:.4f}\tAcc {acc.avg:.3f}'.format(loss=losses, acc=ACC))
    if args.phase == 'train':
        logger.write('[Val] Loss {loss.avg:.4f}\tAcc {acc.avg:.3f}\n'.format(loss=losses, acc=ACC))

    # write results to csv file
    if save_result:
        with open(args.out_csv, 'w') as csv:
            csv.write('video_id,label\n')
            for i in range(len(fnames)):
                csv.write('{},{}\n'.format(fnames[i], int(all_preds[i])))

    return ACC.avg, losses.avg

if __name__ == '__main__':
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=90, help='number of epochs to train (default: 90)')
    parser.add_argument('--lr', type=float, default=1e-3, help='base learning rate (default: 1e-3)')
    parser.add_argument('--step', type=int, default=30, help='learning rate decay step (default: 30)')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate step gamma (default: 0.1)')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size (default: 8)')
    parser.add_argument('--seed', type=int, default=123, help='random seed (default: 123)')
    parser.add_argument('--test_dir', type=str, default='', help='testing image directory')
    parser.add_argument('--checkpoint', type=str, default='', help='pretrained model')
    parser.add_argument('--save_dir', type=str, default='checkpoint/cnn_task2', help='directory to save logfile, checkpoint and output csv')
    parser.add_argument('--out_csv', type=str, default='oulu_val.csv', help='path to output prediction file (csv)')
    parser.add_argument('--data_root', type=str, default='oulu_npu_cropped', help='data root')
    parser.add_argument('--phase', type=str, default='train', help='phase (train/test)')

    args = parser.parse_args()
    print(args)

    main()

