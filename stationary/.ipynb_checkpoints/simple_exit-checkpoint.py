# -*- coding: utf-8 -*-
import torch
import math
import argparse
import datetime
def get_config():
    args = argparse.ArgumentParser()
    # general
    args.add_argument('--lr', default=1e-4, type=float)
    args.add_argument('--sharpness', default=1, type=float)
    args.add_argument('--epsilon', default=1, type=float)
    args.add_argument('--r', default=1, type=float)
    args.add_argument('--noise', default="aniso", type=str)
    return args.parse_args()

dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import sys
from torch import linalg as LA
import math
from torch.nn import functional as F
import copy
def exit_time(config):
    iteration = 0
    dim = 3
    n = 4 
    h_mat = torch.rand((n, dim) , device=device, dtype=dtype) 
    feature_scaler = torch.diag(torch.arange(dim, device=device, dtype=dtype)+1)
    h_mat = config.sharpness*10*torch.matmul(h_mat, feature_scaler)
    h = torch.mean(h_mat, 0)
    # h = config.sharpness * (torch.rand(dim, device=device, dtype=dtype) + 0.5)
    w = (1 / h) * (1 / dim)
    w.requires_grad = True
    y = torch.ones(1, device=device, dtype=dtype)
    origin_w = copy.deepcopy(w)
    while LA.norm(w - origin_w, 2).item() < config.r:
        if config.noise == "aniso":
            w_sgd = copy.deepcopy(w)
            w_sgd.grad = None
        iteration += 1
        fn = (y - torch.dot(h,w))*(y - torch.dot(h,w))
        w.grad = None
        fn.backward()
        if config.noise == "aniso":
            mini_batch = torch.randperm(n)[:1]
            h_sgd = torch.mean(h_mat[mini_batch], 0)
            fn = (y - torch.dot(h_sgd,w_sgd))*(y - torch.dot(h_sgd,w_sgd))
            fn.backward()
        with torch.no_grad():
            if config.noise == "iso":
                noise = torch.randn(dim, device=device, dtype=dtype, requires_grad=False)
            elif config.noise == "aniso":
                noise = w.grad - w_sgd.grad
            w -= config.lr * w.grad + config.epsilon*noise
    return iteration

from mpi4py import MPI
import numpy as np
def main():
    config = get_config()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    data = np.zeros(100)
    data[rank] = exit_time(config)
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
        print(f"{config.epsilon}", end=' ')
        print(f"{config.sharpness}", end=' ')
        print(f"{config.r}", end=' ')
        print(f"{np.mean(totals)}")

if __name__=='__main__':
    main()
