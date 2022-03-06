#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------------------
import datetime
def report(*args):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')+' '+' '.join(map(str,args)).replace('\n',''))

# -------------------------------------------------------------------
import torch
import torchvision
import numpy
from collections import namedtuple
from dataset import CIFAR2
def get_dataset(config):
    transform = []
    transform.append(torchvision.transforms.ToTensor())
    transform.append(torchvision.transforms.Normalize((0.5,), (0.5,)))
    transform = torchvision.transforms.Compose(transform)

    def _get_simplified_dataset(is_train):
        if config["data"] == 'CIFAR10': 
            dataset  = torchvision.datasets.CIFAR10(root='DOWNLOADs', train=is_train,
                                                    download=True,transform=transform)
            x = torch.stack([v[0] for v in dataset])
            y = torch.tensor(dataset.targets)
        elif config["data"] == 'CIFAR2': 
            dataset  = CIFAR2(root='DOWNLOADs', train=is_train)
            x = dataset.data
            y = dataset.targets

        if 0<=config["core"]:
            x = x.cuda()
            y = y.cuda()

        return namedtuple('_','x y n')(x=x, y=y,n=len(y))

    train = _get_simplified_dataset(is_train=True)
    test  = _get_simplified_dataset(is_train=False)
    return namedtuple('_','train test')(train=train, test=test)

# -------------------------------------------------------------------    
import torch
import torchvision
def get_dataloader(config):
    if config["data"] == 'CIFAR10': 
        transform = []
        transform.append(torchvision.transforms.ToTensor())
        transform.append(torchvision.transforms.Normalize((0.5,), (0.5,)))
        transform = torchvision.transforms.Compose(transform)
        train  = torchvision.datasets.CIFAR10(root='DOWNLOADs', train=True,
                                                download=True,transform=transform)
        test  = torchvision.datasets.CIFAR10(root='DOWNLOADs', train=False,
                                                download=True,transform=transform)
    elif config["data"] == 'CIFAR2': 
        train  = CIFAR2(root='DOWNLOADs', train=True)
        test  = CIFAR2(root='DOWNLOADs', train=False)
    train_dataloader = torch.utils.data.DataLoader(train) 
    test_dataloader = torch.utils.data.DataLoader(test) 
    return train_dataloader, test_dataloader

import torch
def set_device(config):
    if 0<=config['core']<torch.cuda.device_count() and torch.cuda.is_available():
        report(f'use GPU; core:{config["core"]}')
        torch.cuda.set_device(config['core'])
    else:
        report('use CPU in this trial')
        config['core'] = -1

import MNIST_model_zoo
import CIFAR10_model_zoo
import CIFAR2_model_zoo
def get_model(config):
    if config["data"] == 'CIFAR10':
        if config["model"]=='LeNet': model = CIFAR10_model_zoo.LeNet()
        elif config["model"]=='MLP':   model = CIFAR10_model_zoo.MLP()
        elif config["model"]=='VGG':   model = CIFAR10_model_zoo.VGG()
    elif config["data"] == 'CIFAR2':
        if config["model"]=='LeNet': model = CIFAR2_model_zoo.LeNet()
        elif config["model"]=='MLP':   model = CIFAR2_model_zoo.MLP()
        elif config["model"]=='VGG':   model = CIFAR2_model_zoo.VGG()
    if 0<=config["core"]: model = model.cuda()
    return model

import torch
import torch.nn.functional as F
def get_loss(config):
    if config["loss"]=='MSE':
        if config["data"] == 'CIFAR10': num_classes = 10
        elif config["data"] == 'CIFAR2': num_classes = 2
        def MSELoss_index(target, index):
            loss = torch.nn.MSELoss()(target, F.one_hot(index, num_classes=num_classes).float())
            return loss
        loss_func = MSELoss_index
    elif config["loss"]=='CE':
        loss_func = torch.nn.CrossEntropyLoss(reduction='mean')
    return loss_func

