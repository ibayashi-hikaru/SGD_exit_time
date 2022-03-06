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
    args.add_argument('--seed',           default=1, type=int)
    args.add_argument('--core',           default=0, type=int)
    args.add_argument('--model',          default='MLP', type=str)
    args.add_argument('--data',           default='CIFAR10', type=str)
    #
    args.add_argument('--target', default="", type=str)
    args.add_argument('--loss', default="MSE", type=str)
    return args.parse_args()


# -------------------------------------------------------------------
import torch
from sharpness.tools import Hessian_trace, Hessian_diag 
import sharpness.Minimum as Minimum
import os, fnmatch
import copy
import pickle
from hessian_eigenthings import compute_hessian_eigenthings
import os
import json
import pickle
def main():
    config = get_config()
    #
    set_device(config)
    set_seed(config)

    train_dataloader, test_dataloader = get_dataloader(config)
    model = get_model(config)
    loss_func = get_loss(config)
    #
    for i in range(5):
        report(f"Trial {i}")
        trial_dir =  config.target+f"/trial_{i}"
        weight_files = fnmatch.filter(os.listdir(trial_dir), '*.weight')
        snapshots = list(map(lambda x: int(x[:5]), weight_files))
        snapshots.sort()
        #
        final_id = snapshots[-1] 
        report(f"Calculate final state")
        status = {}
        status['t'] = final_id
        model.load_state_dict(torch.load(trial_dir + f"/{final_id:05}.weight"))
        #
        status_fn = trial_dir + f"/final.status"
        if os.path.exists(status_fn):
            with open(status_fn, 'rb') as handle:
                saved_status = pickle.load(handle)
        else:
            saved_status = {}
        model.load_state_dict(torch.load(trial_dir + f"/{final_id:05}.weight"))
        num_eigenthings = 1 
        max_eigenval, _ = compute_hessian_eigenthings(model, train_dataloader, loss_func, num_eigenthings)
        status["max_eigenval"] = max_eigenval[0]
        report(f"max_eigenval: {max_eigenval[0]}")

        with open(trial_dir + f"/final.status", 'wb') as handle:
            pickle.dump(status, handle, protocol=pickle.HIGHEST_PROTOCOL)
    report('finish')    


if __name__=='__main__':
    main()

# -------------------------------------------------------------------

