#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fine-tune all layers for bilinear CNN.

This is the second step.
"""


import os
import time

import torch
import torchvision

import cub200
import model

from PIL import Image

import zipfile

from sklearn.metrics import f1_score

torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type(torch.FloatTensor)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = True


__all__ = ['BCNNManager']
__author__ = 'Hao Zhang'
__copyright__ = '2018 LAMDA'
__date__ = '2018-01-11'
__email__ = 'zhangh0214@gmail.com'
__license__ = 'CC BY-SA 3.0'
__status__ = 'Development'
__updated__ = '2018-05-19'
__version__ = '13.1'


class BCNNManager(object):
    """Manager class to train bilinear CNN.

    Attributes:
        _is_all, bool: In the all/fc phase.
        _options, dict<str, float/int>: Hyperparameters.
        _paths, dict<str, str>: Useful paths.
        _net, torch.nn.Module: Bilinear CNN.
        _criterion, torch.nn.Module: Cross-entropy loss.
        _optimizer, torch.optim.Optimizer: SGD with momentum.
        _scheduler, tirch.optim.lr_scheduler: Reduce learning rate when plateau.
        _train_loader, torch.utils.data.DataLoader.
        _test_loader, torch.utils.data.DataLoader.
    """
    def __init__(self, options, paths):
        """Prepare the network, criterion, optimizer, and data.

        Args:
            options, dict<str, float/int>: Hyperparameters.
            paths, dict<str, str>: Useful paths.
        """
        print('Prepare the network and data.')

        # Configurations.
        self._options = options
        self._paths = paths

        # Network.
        if self._paths['pretrained'] is not None:
            self._net = torch.nn.DataParallel(
                model.BCNN(num_classes=200, is_all=True)).cuda()
            self._net.load_state_dict(torch.load(self._paths['pretrained']),
                                                 strict=False)
        else:
            self._net = torch.nn.DataParallel(
                model.BCNN(num_classes=200, is_all=False)).cuda()
        print(self._net)
        self._criterion = torch.nn.CrossEntropyLoss().cuda()

        # Optimizer.
        self._optimizer = torch.optim.SGD(
            self._net.parameters(), lr=self._options['base_lr'],
            momentum=0.9, weight_decay=self._options['weight_decay'])
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer, mode='max', factor=0.1, patience=8, verbose=True,
            threshold=1e-4)

        # Data.
        if self._paths['pretrained'] is not None:
            """train_transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(size=448,
                                                         scale=(0.8, 1.0)),
                # torchvision.transforms.Resize(size=448),
                # torchvision.transforms.RandomCrop(size=448),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225)),

            ])
            test_transforms = torchvision.transforms.Compose([
                # torchvision.transforms.Resize(size=448),
                # torchvision.transforms.CenterCrop(size=448),
                torchvision.transforms.Resize(size=(448, 448)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225)),
            ])
            train_data = cub200.CUB200(
                root=self._paths['cub200'], train=True,
                transform=train_transforms, download=True)
            test_data = cub200.CUB200(
                root=self._paths['cub200'], train=False,
                transform=test_transforms, download=True)"""

            batchsize = 4

            normalize = torchvision.transforms.Normalize(mean=[0.45271412, 0.45271412, 0.45271412],
                                                 std=[0.33165374, 0.33165374, 0.33165374])
            train_transformer = torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.RandomResizedCrop((224),scale=(0.5,1.0)),
                torchvision.transforms.RandomHorizontalFlip(),
            #     transforms.RandomRotation(90),
                # random brightness and random contrast
                torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2),
                torchvision.transforms.ToTensor(),
                normalize
            ])

            val_transformer = torchvision.transforms.Compose([
            #     transforms.Resize(224),
            #     transforms.CenterCrop(224),
                torchvision.transforms.Resize((224,224)),
                torchvision.transforms.ToTensor(),
                normalize
            ])

            print(os.getcwd())
            os.chdir('../')
            print(os.getcwd())
            
            with zipfile.ZipFile('Images-processed/CT_COVID.zip', 'r') as zip_ref:
                zip_ref.extractall('Images-processed')
            with zipfile.ZipFile('Images-processed/CT_NonCOVID.zip', 'r') as zip_ref:
                zip_ref.extractall('Images-processed')

            train_data = CovidCTDataset(root_dir='Images-processed',
                                      txt_COVID='Data-split/COVID/trainCT_COVID.txt',
                                      txt_NonCOVID='Data-split/NonCOVID/trainCT_NonCOVID.txt',
                                      transform= train_transformer)
            test_data = CovidCTDataset(root_dir='Images-processed',
                                      txt_COVID='Data-split/COVID/testCT_COVID.txt',
                                      txt_NonCOVID='Data-split/NonCOVID/testCT_NonCOVID.txt',
                                      transform= val_transformer)
            print(train_data.__len__())
            print(test_data.__len__())
        else:
            train_data = cub200.CUB200ReLU(
                root=self._paths['cub200'], train=True)
            test_data = cub200.CUB200ReLU(
                root=self._paths['cub200'], train=False)

        self._train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batchsize, drop_last=False, shuffle=True)
        self._test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=batchsize, drop_last=False, shuffle=False)

    def train(self):
        """Train the network."""
        print('Training.')
        self._net.train()
        best_acc = 0.0
        best_epoch = None
        print('Epoch\tTrain loss\tTrain acc\tTest acc\tTime')
        for t in range(self._options['epochs']):
            epoch_loss = []
            num_correct = 0
            num_total = 0
            tic = time.time()
            for instances, labels in self._train_loader:
                # Data.
                instances = instances.cuda()
                labels = labels.cuda()

                # Forward pass.
                score = self._net(instances)
                loss = self._criterion(score, labels)

                with torch.no_grad():
                    epoch_loss.append(loss.item())
                    # Prediction.
                    prediction = torch.argmax(score, dim=1)
                    num_total += labels.size(0)
                    num_correct += torch.sum(prediction == labels).item()

                # Backward pass.
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                del instances, labels, score, loss, prediction
            train_acc = 100 * num_correct / num_total
            test_acc = self._accuracy(self._test_loader)
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = t + 1
                #print('*', end=='')
                save_path = os.path.join(
                    self._paths['model'],
                    'bcnn_%s_epoch_%d.pth' % (
                        'all' if self._paths['pretrained'] is not None
                        else 'fc', t + 1))
                torch.save(self._net.state_dict(), save_path)
            toc = time.time()
            print('%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%\t\t%4.2f min' %
                  (t + 1, sum(epoch_loss) / len(epoch_loss), train_acc,
                   test_acc, (toc - tic) / 60))
            self._scheduler.step(test_acc)
        print('Best at epoch %d, test accuaray %4.2f' % (best_epoch, best_acc))

    def _accuracy(self, data_loader):
        """Compute the train/test accuracy.

        Args:
            data_loader: Train/Test DataLoader.

        Returns:
            Train/Test accuracy in percentage.
        """
        with torch.no_grad():
            self._net.eval()
            num_correct = 0
            num_total = 0
            for instances, labels in data_loader:
                # Data.
                instances = instances.cuda()
                labels = labels.cuda()

                # Forward pass.
                score = self._net(instances)

                # Predictions.
                prediction = torch.argmax(score, dim=1)
                num_total += labels.size(0)
                num_correct += torch.sum(prediction == labels).item()
            self._net.train()  # Set the model to training phase
        print(str(num_correct), str(num_total))
        fcscore = f1_score(num_correct, num_total, average='binary')
        print('F1: ' + str(fcscore))
        return 100 * num_correct / num_total

def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data

class CovidCTDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, txt_COVID, txt_NonCOVID, transform=None):
        """
        Args:
            txt_path (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        File structure:
        - root_dir
            - CT_COVID
                - img1.png
                - img2.png
                - ......
            - CT_NonCOVID
                - img1.png
                - img2.png
                - ......
        """
        self.root_dir = root_dir
        self.txt_path = [txt_COVID,txt_NonCOVID]
        self.classes = ['CT_COVID', 'CT_NonCOVID']
        self.num_cls = len(self.classes)
        self.img_list = []
        for c in range(self.num_cls):
            cls_list = [[os.path.join(self.root_dir,self.classes[c],item), c] for item in read_txt(self.txt_path[c])]
            self.img_list += cls_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_list[idx][0]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        """sample = {'img': image,
                  'label': int(self.img_list[idx][1])}"""
        return image, int(self.img_list[idx][1])


def main(args = None):
    """The main function."""
    import argparse
    parser = argparse.ArgumentParser(
        description='Train mean field bilinear CNN on CUB200.')
    parser.add_argument('--base_lr', dest='base_lr', type=float, required=True,
                        help='Base learning rate for training.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        required=True, help='Batch size.')
    parser.add_argument('--epochs', dest='epochs', type=int, required=True,
                        help='Epochs for training.')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float,
                        required=True, help='Weight decay.')
    parser.add_argument('--pretrained', dest='pretrained', type=str,
                        required=False, help='Pre-trained model.')
    args = parser.parse_args(args)
    if args.base_lr <= 0:
        raise AttributeError('--base_lr parameter must >0.')
    if args.batch_size <= 0:
        raise AttributeError('--batch_size parameter must >0.')
    if args.epochs < 0:
        raise AttributeError('--epochs parameter must >=0.')
    if args.weight_decay <= 0:
        raise AttributeError('--weight_decay parameter must >0.')

    project_root = os.popen('pwd').read().strip()
    options = {
        'base_lr': args.base_lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
    }
    paths = {
        'cub200': os.path.join(project_root, 'data', 'cub200'),
        'aircraft': os.path.join(project_root, 'data', 'aircraft'),
        'model': os.path.join(project_root, 'model'),
        'pretrained': (os.path.join(project_root, 'model', args.pretrained)
                       if args.pretrained else None),
    }
    for d in paths:
        print(str(paths[d]))
        if d == 'pretrained':
            assert paths[d] is None or os.path.isfile(paths[d])
        else:
            assert os.path.isdir(paths[d])

    manager = BCNNManager(options, paths)
    manager.train()


if __name__ == '__main__':
    main()
