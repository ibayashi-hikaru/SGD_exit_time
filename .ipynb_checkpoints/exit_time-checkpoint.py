#!/usr/bin/env python
# coding: utf-8

"""
import torch
torch.set_num_threads(7)
torch.set_num_interop_threads(7)
torch.backends.cudnn.benchmark = True
#"""
from utility import *
# ------------------------------------------------------
import argparse
import datetime
def get_config():
    args = argparse.ArgumentParser()
    # general
    args.add_argument('--core',           default=0, type=int)
    args.add_argument('--model',          default='MLP', type=str)
    args.add_argument('--data',           default='AVILA2', type=str)
    args.add_argument('--seed',           default=-1, type=int)
    # optim 
    args.add_argument('--batch_size',     default=2**7, type=int)
    args.add_argument('--lr',             default=1e-2, type=float)
    args.add_argument('--epsilon',             default=1e-2, type=float)
    args.add_argument('--loss', default="MSE", type=str)
    args.add_argument('--sharpness', default=1, type=float)
    args.add_argument('--noise', default="aniso", type=str)
    args.add_argument('--r', default=0.005, type=float)
    return args.parse_args()

# -------------------------------------------------------------------
import torch
import math
from sharpness.tools import Hessian_trace, Hessian_diag 
import sharpness.Minimum as Minimum
import os
import copy
def exit_time(dataset, model, config):
    origin = copy.deepcopy(model)
    loss_func = get_loss(config)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, weight_decay=1e-5, momentum=0.9)

    def step():
        model.train()
        measure = {'loss':0,'accuracy':0}
        random_index = torch.randperm(dataset.train.n)
        gd_model = copy.deepcopy(model)
        gd_model.zero_grad()
        # Full batch gradient
        x = dataset.train.x
        y = dataset.train.y
        o = gd_model(x)
        loss_func(o,y).backward()
        if config.noise == "aniso":
            sgd_model = copy.deepcopy(model)
            sgd_model.zero_grad()
            # Mini batch gradient
            mini_batch = random_index[:config.batch_size]
            x = dataset.train.x[mini_batch]
            y = dataset.train.y[mini_batch]
            o = sgd_model(x)
            loss_func(o,y).backward()
        if config.noise == "iso":
            with torch.no_grad():
                for p, gd_p in zip(model.parameters(), gd_model.parameters()):
                    if p.requires_grad:
                        p -= config.lr*gd_p.grad
                        p += config.epsilon*torch.randn(p.shape).cuda()
        elif config.noise == "aniso":
            with torch.no_grad():
                for p, gd_p, sgd_p in zip(model.parameters(), gd_model.parameters(), sgd_model.parameters()):
                    if p.requires_grad:
                        p -= config.lr*gd_p.grad
                        p -= config.epsilon*(sgd_p.grad - gd_p.grad)


    def l2_deviation():
        l2_distance = 0 
        for origin_param, perturbed_param in zip(origin.parameters(), model.parameters()) :
            deviation = perturbed_param.data - origin_param.data
            l2_distance += torch.norm(deviation, 2).item()
        return l2_distance

    iteration = 0
    while l2_deviation() < config.r:
        iteration += 1
        step()
    return iteration

# -------------------------------------------------------------------    
import os
import time
import json
import shutil
from mpi4py import MPI
import numpy as np
def main():
    config = get_config()
    set_device(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = get_dataset(config)
    if config.noise == "iso":
        config.batch_size = dataset.train.n
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    data = np.zeros(100)
    model = get_model(config)
    model.load_state_dict(torch.load(f"./test_data/Trained.weight",map_location=device))

    if config.sharpness != 1:
        param = model.get_param()
        param["layer1.weight"] /= math.sqrt(config.sharpness)
        model.set_param(param)
    data[rank] = exit_time(dataset, model, config)
    comm.Barrier()
    if comm.rank==0:
        # only processor 0 will actually get the data
        totals = np.zeros_like(data)
    else:
        totals = None
    comm.Reduce(
        [data, MPI.DOUBLE],
        [totals, MPI.DOUBLE],
        op = MPI.SUM,
        root = 0
    )
    if rank == 0:
        print(f"{config.batch_size}", end=' ')
        print(f"{config.epsilon}", end=' ')
        print(f"{config.sharpness}", end=' ')
        print(f"{config.r}", end=' ')
        print(f"{np.mean(totals)}")

if __name__=='__main__':
    main()
        #

# -------------------------------------------------------------------

