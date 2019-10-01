import os
import argparse
import collections
import sys

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(__file__))
    import src
    __package__ = "src"

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from .models.RetinaNet import RetinaNet

from .datasets.coco_loader import CocoDataset
from .datasets.csv_loader import CSVDataset
from .datasets.data_utils import collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer

from .evaluates import coco_eval
from .evaluates import csv_eval

from tensorboardX import SummaryWriter

print('CUDA available: {}'.format(torch.cuda.is_available()))

def parse_args(args):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')

    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--coco_version', help='COCO dataset version')
    parser.add_argument('--coco_train', help='Name of train set', type=str, default='train')
    parser.add_argument('--coco_val', help='Name of train set', type=str, default='val')

    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--save_name', help='Name of saved model name', type=str)
    parser.add_argument('--resume_epoch', help='Epoch where we resume', type=int, default=0)
    parser.add_argument('--model', help='Path to pretrained model (.pt) file.')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)

    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=1)
    parser.add_argument('--num_worker', help='Number of workers', type=int, default=0)
    parser.add_argument('--lr', help='Learning Rate', type=float, default=1e-5)
    parser.add_argument('--clip_grad_norm', help='Clip Grad Norm Parameter', type=float, default=1e-3)

    parser.add_argument('--testOnly', help='Test only', action='store_true')
    parser.add_argument('--use_bbox_result', help='Load bbox_result.json in Evaluation Section', action='store_true')
    parser.add_argument('--save_bbox_only', help='only Save bbox_result.json in Evaluation Section', action='store_true')
    parser.add_argument('--top_k', help='nms top_k', default=300)

    parser = parser.parse_args(args)
    return parser

def dataset_loader(parser):
    if parser.dataset == 'coco':
        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')
        dataset_train = CocoDataset(parser.coco_path, set_name=parser.coco_train+parser.coco_version, transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name=parser.coco_val+parser.coco_version, transform=transforms.Compose([Normalizer(), Resizer()]))

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
    dataloader_train = DataLoader(dataset_train, num_workers=parser.num_worker, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=parser.num_worker, collate_fn=collater, batch_sampler=sampler_val)

    return dataloader_train, dataloader_val, dataset_train, dataset_val

def load_model(parser, num_classes):
    retinanet = RetinaNet(num_classes=num_classes, resnet_size=parser.depth, pretrained=True, top_k=parser.top_k)

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

    return retinanet

def eval_model(parser, dataset_val, model):
    if parser.dataset == 'coco':
        print('\nEvaluating dataset')
        coco_eval.evaluate_coco(dataset_val, model, parser)

    elif parser.dataset == 'csv' and parser.csv_val is not None:
        print('\nEvaluating dataset')
        mAP = csv_eval.evaluate(dataset_val, model)

def main(args=None):
    parser = parse_args(args)
    log_dir = 'log_'+parser.save_name
    summary = SummaryWriter(log_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists('./checkpoints/'):
        os.makedirs('./checkpoints')

    ########################################################################
    # dataset & model
    ########################################################################
    dataloader_train, dataloader_val, dataset_train, dataset_val = dataset_loader(parser)
    retinanet = load_model(parser, dataset_train.num_classes())

    if parser.testOnly:
        eval_model(parser, dataset_val, retinanet)
        return

    ########################################################################
    # optimizer
    ########################################################################
    optimizer = optim.Adam(retinanet.parameters(), lr=parser.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    loss_hist = collections.deque(maxlen=500)

    ########################################################################
    # epoch
    ########################################################################
    print('Num training images: {}'.format(len(dataset_train)))
    for epoch_num in range(parser.resume_epoch, parser.epochs):
        retinanet.train()
        retinanet.module.freeze_bn()
        epoch_loss = []

        ########################################################################
        # Training
        ########################################################################
        for iter_num, data in enumerate(dataloader_train):
            optimizer.zero_grad()

            cls_loss, reg_loss = retinanet([data['img'].cuda().float(), data['annot'].cuda().float()])
            loss = cls_loss + reg_loss
            if bool(loss == 0): continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), parser.clip_grad_norm)
            optimizer.step()

            loss_hist.append(float(loss))
            epoch_loss.append(float(loss))

            sys.stdout.write('\r')
            sys.stdout.write('Epoch: {} | Iteration: {}/{} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                                epoch_num, iter_num, len(dataloader_train),
                                float(cls_loss),
                                float(reg_loss),
                                np.mean(loss_hist))
                            )
            sys.stdout.flush()
            sys.stdout.write('\r')

            if iter_num % 10 == 0:
                curr_iter = epoch_num*len(dataloader_train) + iter_num
                summary.add_scalar('Loss', np.mean(loss_hist), curr_iter)
                summary.add_scalar('Loss/regression_loss', reg_loss.mean().item(), curr_iter)
                summary.add_scalar('Loss/classification_loss', cls_loss.mean().item(), curr_iter)

            del cls_loss
            del reg_loss

        ########################################################################
        # Validation
        ########################################################################
        eval_model(parser, dataset_val, retinanet)
        scheduler.step(np.mean(epoch_loss))
        torch.save(retinanet.state_dict(), './checkpoints/{}_{}.pt'.format(parser.save_name, epoch_num+1))

    retinanet.eval()
    torch.save(retinanet, 'checkpoints/{}_final.pt'.format(parser.save_name))

if __name__ == '__main__':
    main()
