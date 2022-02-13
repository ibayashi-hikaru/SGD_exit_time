#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------------------
import datetime
def report(rank, *args):
    if rank == 0:
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')+' '+' '.join(map(str,args)).replace('\n',''), flush=True)

# -------------------------------------------------------------------
import torch
import torchvision
import numpy
from collections import namedtuple
from dataset import AVILA2
def get_dataset(config):
    transform = []
    transform.append(torchvision.transforms.ToTensor())
    transform.append(torchvision.transforms.Normalize((0.5,), (0.5,)))
    transform = torchvision.transforms.Compose(transform)

    def _get_simplified_dataset(is_train):
        if config["data"] == 'MNIST':
            dataset  = torchvision.datasets.MNIST(root='DOWNLOADs', train=is_train,
                                                  download=True,transform=transform)
            x = torch.stack([v[0] for v in dataset])
            y = dataset.targets
        elif config["data"] == 'CIFAR10': 
            dataset  = torchvision.datasets.CIFAR10(root='DOWNLOADs', train=is_train,
                                                    download=True,transform=transform)
            x = torch.stack([v[0] for v in dataset])
            y = torch.tensor(dataset.targets)
        elif config["data"] == 'AVILA2': 
            dataset  = AVILA2(root='DOWNLOADs', train=is_train)
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
import random
import numpy
import torch
import time
def set_seed(config):
    if config["seed"]<0:
        config["seed"] = int(time.time())
    random.seed(config["seed"])
    numpy.random.seed(config["seed"])
    torch.manual_seed(config["seed"])    

import torch
def set_device(config):
    if 0<=config["core"]<torch.cuda.device_count() and torch.cuda.is_available():
        # report(f'use GPU; core:{config["core"]}')
        torch.cuda.set_device(config["core"])
    else:
        # report('use CPU in this trial')
        config["core"] = -1

import AVILA2_model_zoo
def get_model(config, sharpness):
    if config["data"] == 'AVILA2':
        if config["model"]=='quad_func':
            model = AVILA2_model_zoo.quad_func(config["num_dim"], sharpness)
        elif config["model"]=='styblinski_tang_func':
            model = AVILA2_model_zoo.styblinski_tang_func(config["num_dim"], sharpness)
        elif config["model"]=='MLP':
            model = AVILA2_model_zoo.MLP(config["num_dim"], sharpness)
    if 0<=config["core"] and torch.cuda.is_available():
        model = model.cuda()
    return model

import torch
import torch.nn.functional as F
def get_loss(config):
    if config["loss"]=='MSE':
        if config["data"] == 'AVILA2': num_classes = 2
        else:  num_classes = 10
        def MSELoss_index(target, index):
            loss = torch.nn.MSELoss()(target, F.one_hot(index, num_classes=num_classes).float())
            return loss
        loss_func = MSELoss_index
    elif config["loss"]=='CE':
        loss_func = torch.nn.CrossEntropyLoss(reduction='mean')
    return loss_func

from torch.autograd.functional import hessian
from torch.autograd import grad
from numpy import linalg as LA
import itertools
def eigen_vals(model, dataset, loss_func):
    p_num = 0
    for p in model.parameters():
        if p.requires_grad:
            p_num += p.contiguous().view(-1).size()[0]
    train_index = 1
    x = dataset.train.x
    y = dataset.train.y
    o = model(x)
    loss = loss_func(o,y)
    grad1st = grad(loss, itertools.islice(model.parameters(), train_index, None), create_graph=True)
    cnt = 0
    for g in grad1st:
        g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
        cnt = 1
    hessian = torch.zeros(p_num, p_num).cuda()
    for idx in range(p_num):
        grad2rd = grad(g_vector[idx], itertools.islice(model.parameters(), train_index, None), create_graph=True)
        cnt = 0
        for g in grad2rd:
            g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
            cnt = 1
        hessian[idx] = g2
    H = hessian.cpu().data.numpy()
    eigens = LA.eig(H)[0].real
    eigens.sort(axis=0)
    return eigens
