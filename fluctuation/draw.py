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
def get_config():
    args = argparse.ArgumentParser()
    # general
    args.add_argument('--seed',           default=1, type=int)
    args.add_argument('--core',           default=0, type=int)
    args.add_argument('--model',          default='MLP', type=str)
    args.add_argument('--data',           default='CIFAR10', type=str)
    #
    args.add_argument('--target', default="", type=str)
    return args.parse_args()

# -------------------------------------------------------------------
from math import log10, floor
def round_sig(x, sig=3):
    if x == 0: return 0
    return round(x, sig-int(floor(log10(abs(x))))-1)

# -------------------------------------------------------------------
import os
import matplotlib.pyplot as plt
import matplotlib
import uuid
from numpy import mean, std
def draw(trails, config, target_dir):
    
    linestyles = ['solid','dotted','dashed','dashdot', (0, (1, 10))]

    final_status = {'test_accuracy'    : [],
                    'train_accuracy'   : [],
                    'gen_error'        : [],
                    'sharpness'        : []}
    for i, (trail, linestyle) in enumerate(zip(trails, linestyles)):
        t = [status['t'] for status in trail]
        # Accuracy 
        test_accuracy = [status['test_accuracy'] for status in trail]
        # ax1.plot(t, test_accuracy,
        #          linestyle=linestyle, lw=1.0, color='red',
        #          label="Test Accuracy" if i==0 else None)
        train_accuracy = [status['train_accuracy'] for status in trail]
        # ax1.plot(t, train_accuracy,
        #         linestyle=linestyle,lw=1.0, color='black',
        #         label="Train Accuracy" if i==0 else None)
        # Sharpness
        sharpness = [status['sharpness'] for status in trail]
        # ax2.plot(t, sharpness,
        #          linestyle=linestyle, lw=1.0, color="blue",
        #          label="sharpness" if i == 0 else None)
        # l2_norm
        l2_norm = [status['l2_norm'] for status in trail]
        # ax3.plot(t, l2_norm,
        #          linestyle=linestyle, lw=1.0, color="magenta",
        #          label="l2 norm" if i ==0 else None)
        # keskar sharpness
        # keskar_sharpness = [status['keskar_sharpness'] for status in trail]
        # ax4.plot(t, keskar_sharpness,
        #         linestyle=linestyle, lw=1.0, color="blue",
        #         label="keskar sharpness" if i==0 else None)
        #
        final_status['test_accuracy'].append(test_accuracy[-1])
        final_status['train_accuracy'].append(train_accuracy[-1])
        final_status['gen_error'].append(train_accuracy[-1] - test_accuracy[-1])
        final_status['sharpness'].append(sharpness[-1])
    #
    # ax1.set_xlabel('Epochs')
    # ax1.set_ylabel('Accuracy')
    # # ax1.set_ylim([0.4, 1.0])
    # ax1.legend()
    # ax2.set_xlabel('Epochs')
    # ax2.set_ylabel('Sharpness')
    # # ax2.set_ylim([0, 100])
    # ax2.legend()
    # ax3.set_xlabel('Epochs')
    # ax3.set_ylabel('L2 norm')
    # # ax3.set_ylim([15, 70])
    # ax3.legend()
    # ax4.set_xlabel('Epochs')
    # ax4.set_ylabel('Keskar Sharpness')
    # # ax4.set_ylim([0, 400])
    # ax4.legend()
    # #
    # ax6.set_xlabel('Epochs')
    # ax6.set_ylabel('Sharpness Ratio')
    # # ax6.set_ylim([0, 400])
    # ax6.legend()

    # mean_list = []
    # std_list = []
    # name_list = []
    # for measure_name, measure_list in final_status.items():
    #     name_list.append(measure_name)
    #     mean_list.append(round_sig(mean(measure_list),3))
    #     std_list.append(round_sig(std(measure_list),2))
    # gs = ax7.get_gridspec()
    # ax7.remove()
    # ax8.remove()
    # axbig = fig.add_subplot(gs[3, :])
    # column_names = name_list
    # row_names=['mean', 'std']
    # values=[mean_list, std_list]
    # axbig.axis('off') 
    # table = axbig.table(cellText=values,colLabels=column_names,rowLabels=row_names,loc='center')
    # table.set_fontsize(15)
    # table.scale(1, 2)  # may help

    # fn = f'trajectory.pdf'
    # report(f'Saving to {fn}')
    # fig.savefig(fn)

    font = {'size' : 22}
    matplotlib.rc('font', **font)
    figure = plt.gcf()
    figure.set_size_inches(12, 8)
    plt.xlabel('Steps')
    plt.ylabel('Sharpness')
    plt.plot(t, sharpness, linestyle='--', marker='o', linewidth=2, markersize=5)
    plt.savefig("fluctuation.pdf",dpi=100)

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
    trails = []
    for i in range(1):
        report(f"Trial {i}")
        trial_dir = f"results/trial_{i}"
        if not os.path.exists(trial_dir): continue
        weight_files = fnmatch.filter(os.listdir(trial_dir), '*.weight')
        snapshots = list(map(lambda x: int(x[:5]), weight_files))
        snapshots.sort()
        trail = []
        for snapshot_id in snapshots:
            if os.path.exists(trial_dir + f"/{snapshot_id:05}.status"):
                report(f"Epoch: {snapshot_id}")
                with open(trial_dir + f"/{snapshot_id:05}.status", 'rb') as handle:
                    measures = pickle.load(handle)
            else:
                break

            trail.append(measures)
        trails.append(trail)

    report('start draw')    
    draw(trails, config, "results")
    report('finish')    


if __name__=='__main__':
    main()

# -------------------------------------------------------------------

