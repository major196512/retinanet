import os
import argparse
import collections
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from models.RetinaNet import RetinaNet

from datasets.coco_loader import CocoDataset
from datasets.csv_loader import CSVDataset
from datasets.data_utils import collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer

from evaluates import coco_eval
from evaluates import csv_eval

from tensorboardX import SummaryWriter

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):

    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')

    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--coco_class', help='COCO dataset class')
    parser.add_argument('--coco_train', help='Name of train set', type=str, default='train')
    parser.add_argument('--coco_val', help='Name of train set', type=str, default='val')

    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--save_name', help='Name of saved model name', type=str)
    parser.add_argument('--log_dir', help='Path to log folder', type=str)
    parser.add_argument('--model', help='Path to pretrained model (.pt) file.')

    parser.add_argument('--resume_epoch', help='Epoch where we resume', type=int, default=0)
    parser.add_argument('--lr', help='Learning Rate', type=float, default=1e-5)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=1)
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)

    parser = parser.parse_args(args)
    summary = SummaryWriter(parser.log_dir)

    if not os.path.exists(parser.log_dir):
        os.makedirs(parser.log_dir)
    if not os.path.exists('./checkpoints/'):
        os.makedirs('./checkpoints')

	############################################################################
    # 1> Load Dataset
    ############################################################################
    if parser.dataset == 'coco':
        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')
        dataset_train = CocoDataset(parser.coco_path, set_name=parser.coco_train+parser.coco_class, transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name=parser.coco_val+parser.coco_class, transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':
        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')
        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')
        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

	############################################################################
    # 2> Load Model
    ############################################################################
    retinanet = RetinaNet(num_classes=dataset_train.num_classes(), resnet_size=parser.depth, pretrained=True)

    if parser.resume_epoch > 0:
        retinanet = torch.nn.DataParallel(retinanet)
        retinanet.load_state_dict(torch.load('./checkpoints/{}_{}.pt'.format(parser.save_name, parser.resume_epoch)))
    elif parser.model:
        retinanet.load_state_dict(torch.load(parser.model), strict=False) # Load pre-trained retinanet
        retinanet = torch.nn.DataParallel(retinanet)
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device('cuda')
        retinanet = retinanet.to(device)
    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=parser.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    loss_hist = collections.deque(maxlen=500)

    ############################################################################
    # 3> Training
    ############################################################################
    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(parser.resume_epoch, parser.epochs):
        retinanet.train()
        retinanet.module.freeze_bn()
        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            optimizer.zero_grad()

            classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot'].cuda().float()])

            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            loss = classification_loss + regression_loss

            if bool(loss == 0):
                continue

            loss.backward()
            #torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.001)
            optimizer.step()

            loss_hist.append(float(loss))
            epoch_loss.append(float(loss))

            sys.stdout.write('\r')
            sys.stdout.write('Epoch: {} | Iteration: {}/{} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(epoch_num, iter_num, len(dataloader_train), float(classification_loss), float(regression_loss), np.mean(loss_hist)))
            sys.stdout.flush()
            sys.stdout.write('\r')

            if iter_num % 10 == 0:
                curr_iter = epoch_num*len(dataloader_train) + iter_num
                summary.add_scalar('Loss', np.mean(loss_hist), curr_iter)
                summary.add_scalar('Loss/regression_loss', regression_loss.mean().item(), curr_iter)
                summary.add_scalar('Loss/classification_loss', classification_loss.mean().item(), curr_iter)

            del classification_loss
            del regression_loss

        ########################################################################
        # 4> Validation
        ########################################################################
        if parser.dataset == 'coco':
            print('\nEvaluating dataset')
            coco_eval.evaluate_coco(dataset_val, retinanet)

        elif parser.dataset == 'csv' and parser.csv_val is not None:
            print('\nEvaluating dataset')
            mAP = csv_eval.evaluate(dataset_val, retinanet)

        scheduler.step(np.mean(epoch_loss))
        torch.save(retinanet.state_dict(), './checkpoints/{}_{}.pt'.format(parser.save_name, epoch_num+1))

    retinanet.eval()
    torch.save(retinanet, 'checkpoints/{}_final.pt'.format(parser.save_name))

if __name__ == '__main__':
    main()
