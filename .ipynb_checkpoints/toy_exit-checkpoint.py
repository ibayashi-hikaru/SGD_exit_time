import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float
def get_exit_time(config, lr, sharpness):
    sqrt_s = np.sqrt(sharpness)
    exit_time_arr = np.zeros(config['exit_trial_num'])
    for sample_id in range(config['exit_trial_num']):
        # Define Origin to a minimum
        # Styblinski-Tang
        # init_params = (-2.903534/sqrt_s) * torch.ones((1, config['num_dim']), device=device, dtype=dtype)
        # params = (-2.903534/sqrt_s) * torch.ones((1, config['num_dim']), device=device, dtype=dtype)
        # Quadratic func
        init_params = (0/sqrt_s) * torch.ones((1, config['num_dim']), device=device, dtype=dtype)
        params = (0/sqrt_s) * torch.ones((1, config['num_dim']), device=device, dtype=dtype)
        #
        params.requires_grad=True
        exit_time = 0
        while(torch.norm(params-init_params) < config['r']):
            # Styblinski-Tang
            # func = 0.5 * ((sqrt_s*params) ** 4 - 16 * (sqrt_s*params) ** 2 + 5 * (sqrt_s*params)).sum(dim=1)
            # Quadratic func
            func = ((sqrt_s*params) ** 2).sum(dim=1)
            func.backward()
            with torch.no_grad():
                params -= lr*params.grad
                params -= 0.1*torch.randn(config['num_dim'], device=device, dtype=dtype)
            exit_time += 1
        exit_time_arr[sample_id] = exit_time
    return np.mean(exit_time_arr), np.std(exit_time_arr)
from utility import *
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import optuna
from mpi4py import MPI
%matplotlib inline
def main():
    config = {}
    config['core'] = 0
    config['seed'] = -1
    config['num_dim'] = 2
    config['sharpness_min'] = 1
    config['sharpness_interval'] = 1
    config['r'] = 1
    config['lr_min'] = 0.005
    config['lr_interval'] = 0.1
    config['exit_trial_num'] = 500
    config['interval_sample'] = 10

    set_device(config)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # Sharpess
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    data = np.zeros(1000)
    sharpness_arr = np.linspace(config['sharpness_min'], 
                                 config['sharpness_min']+config['sharpness_interval'],
                                 config['interval_sample'])
    exit_time_arr = np.zeros((0,1))
    std_arr = np.zeros((0,1))
    for sharpness in sharpness_arr:
        mean, std = get_exit_time(config, config['lr_min'], sharpness)
        exit_time_arr = np.append(exit_time_arr, mean)
        std_arr = np.append(std_arr, std)
    ax1.set_xlabel("sharpness")
    ax1.set_ylabel("exit time")
    ax1.errorbar(sharpness_arr, exit_time_arr, yerr=std_arr, fmt='.k') 
    ax1.plot(sharpness_arr, exit_time_arr) 
    # Learning rate
    lr_arr = np.linspace(config['lr_min'],
                          config['lr_min']+config['lr_interval'],
                          config['interval_sample'])
    exit_time_arr = np.zeros((0,1))
    std_arr = np.zeros((0,1))
    for lr in lr_arr:
        mean, std = get_exit_time(config, config['sharpness_min'], lr)
        exit_time_arr = np.append(exit_time_arr, mean)
        std_arr = np.append(std_arr, std)
    ax2.set_xlabel("learning rate")
    ax2.set_ylabel("exit time")
    ax2.errorbar(lr_arr, exit_time_arr, yerr=std_arr, fmt='.k') 
    ax2.plot(lr_arr, exit_time_arr)
    # A = np.vstack([x, np.ones(len(x))]).T
    # lr_m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    

if __name__=='__main__':
    main()