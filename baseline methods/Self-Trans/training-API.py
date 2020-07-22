"""#!/usr/bin/env python
# coding: utf-8
"""
#import torchvision.models as models
import argparse
import main_moco
import sys
"""
Training API for tested models
"""
model_names = {
        # Model from original authors
        #"Dense169_self_trans":  models.densenet169(pretrained=True),

        # Moel from Team5
        "team 5 model": "file path",

        # Model from Team6
        "blinear-cnn-pretrained": "blinear-cnn-pretrained file path"
    }


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--train-script', '--ts', default="main_moco.py",
                    help='Name of python file to run for training')


parser.add_argument('--data',  default='LUNA', type=str,
                    help='path to dataset')
parser.add_argument('-a', '--arch', default='resnet50',type=str,
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', default=True,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--save-epoch',default=40,type=int)
parser.add_argument('--save-path',default='LUNA_resnet50_128_imagenet',type=str)

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=128, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', default=True,
                    help='use mlp head')
parser.add_argument('--aug-plus', default=True,
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', default=True,
                    help='use cosine lr schedule')

args = parser.parse_args()


def main():

    newArgs = []
    for arg in vars(args):
        if arg != "train_script"
            attr = getattr(args, arg)

            if attr != None:
                attr = "{}".format(attr)
                newArgs.append("--{}".format(arg))
                newArgs.append(attr)


    if args.train_script == "main_moco.py":
        main_moco.main(newArgs)


if __name__ == "__main__":
    main()