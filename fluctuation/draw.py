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
import uuid

# -------------------------------------------------------------------
import os
import matplotlib.pyplot as plt
import matplotlib
import uuid
from numpy import mean, std
def draw(trail, config, target_dir):
    t = [status['t'] for status in trail]
    test_accuracy = [status['test_accuracy'] for status in trail]
    train_accuracy = [status['train_accuracy'] for status in trail]
    sharpness = [status['sharpness'] for status in trail]
    l2_norm = [status['l2_norm'] for status in trail]
    #
    font = {'size' : 22}
    matplotlib.rc('font', **font)
    figure = plt.gcf()
    figure.set_size_inches(12, 8)
    plt.xlabel('Steps')
    plt.ylabel('Sharpness')
    plt.plot(t, sharpness, linestyle='--', marker='o', linewidth=2, markersize=5)
    plt.savefig("fluctuation.pdf",dpi=100)
    #
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(16, 12))
    # Accuracy 
    ax1[0].plot(t, test_accuracy, lw=1.0, color='red', label="Test Accuracy")
    ax1[0].plot(t, train_accuracy,lw=1.0, color='black', label="Train Accuracy")
    ax1[1].plot(t, sharpness, lw=1.0, color="blue", label="sharpness" )
    ax2[0].plot(t, l2_norm, lw=1.0, color="magenta", label="l2 norm")
    #
    #
    ax1[0].set_xlabel('steps')
    ax1[0].set_ylabel('Accuracy')
    ax1[1].set_xlabel('steps')
    ax1[1].set_ylabel('Sharpness')
    ax2[0].set_xlabel('step')
    ax2[0].set_ylabel('L2 norm')

    figure.set_size_inches(12, 8)
    fn = f'summary.pdf'
    fig.savefig(fn)


# -------------------------------------------------------------------    
import os, fnmatch
import json
import pickle
from hessian_eigenthings import compute_hessian_eigenthings
def main():
    config_fn = "./config.json"
    with open(config_fn) as json_file:
        config = json.load(json_file)
    #
    set_device(config)

   #
    trial_dir = f"results/trial_{0}"
    weight_files = fnmatch.filter(os.listdir(trial_dir), '*.weight')
    snapshots = list(map(lambda x: int(x[:5]), weight_files))
    snapshots.sort()
    trail = []
    for snapshot_id in snapshots:
        if os.path.exists(trial_dir + f"/{snapshot_id:05}.status"):
            report(f"step {snapshot_id}")
            with open(trial_dir + f"/{snapshot_id:05}.status", 'rb') as handle:
                measures = pickle.load(handle)
        else:
            break
        trail.append(measures)
    report('start draw')    
    draw(trail, config, "results")
    report('finish')    


if __name__=='__main__':
    main()

# -------------------------------------------------------------------

