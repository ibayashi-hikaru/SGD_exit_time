from utility import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float
#
dataset = None
#
import torch
import copy
def get_exit_time(config, lr, sharpness, batch_size, r):
    model = get_model(config, sharpness)
    model.init_params()
    init_model = copy.deepcopy(model)
    #
    def distance(init_model, model):
        with torch.no_grad():
            distance = 0
            for init_param, param in zip(init_model.parameters(), model.parameters()):
                distance += torch.norm(param - init_param)
        return distance.item()

    exit_time = 0
    while(distance(init_model, model) < r):
        if config["optim"] == "SGD":
            data_size = dataset.train.x.size()[0]
            shuffled_data = dataset.train.x[torch.randperm(data_size)]
            mini_batch = shuffled_data[:batch_size,:]
            model(mini_batch).backward()
            model.update(lr)
        elif config["optim"] == "SGLD":
            model(dataset.train.x).backward()
            model.update(lr)
            model.perturb(lr)
        else:
            assert False
        exit_time += 1
    return  exit_time

def get_lr_vs_exit_time(config, comm):
    lr_arr = np.linspace(config['lr_min'],
                          config['lr_min']+config['lr_interval'],
                          config['interval_sample'])
    comm_size = comm.Get_size()
    rank = comm.Get_rank()
    exit_time_arr = np.zeros((0,1))
    log_exit_time_arr = np.zeros((0,1))
    std_arr = np.zeros((0,1))
    log_std_arr = np.zeros((0,1))
    #
    stored_exit_time = np.zeros(config['exit_trial_num'])
    for lr in lr_arr:
        sample_id = rank
        while sample_id < config['exit_trial_num']:
            stored_exit_time[sample_id] = get_exit_time(config,
                                                        lr,
                                                        config['sharpness_min'],
                                                        config['batch_size_min'],
                                                        config['r_min'])
            sample_id += comm_size
        comm.Barrier()
        if rank==0:
            reduced_exit_time = np.zeros_like(stored_exit_time) 
        else:
            reduced_exit_time = None 
        comm.Reduce( [stored_exit_time, MPI.DOUBLE],
                     [reduced_exit_time, MPI.DOUBLE],
                     op = MPI.SUM, root = 0)
        if rank==0:
            exit_time_arr = np.append(exit_time_arr, np.mean(reduced_exit_time))
            std_arr = np.append(std_arr, np.std(reduced_exit_time))
            log_exit_time_arr = np.append(log_exit_time_arr, np.mean(np.std(np.log(reduced_exit_time))))
            log_std_arr = np.append(log_std_arr, np.std(np.log(reduced_exit_time)))
    return (lr_arr, exit_time_arr, std_arr, log_exit_time_arr, log_std_arr)

def get_sharpness_vs_exit_time(config, comm):
    sharpness_arr = np.linspace(config['sharpness_min'], 
                                config['sharpness_min']+config['sharpness_interval'],
                                config['interval_sample'])
    comm_size = comm.Get_size()
    rank = comm.Get_rank()
    exit_time_arr = np.zeros((0,1))
    log_exit_time_arr = np.zeros((0,1))
    std_arr = np.zeros((0,1))
    log_std_arr = np.zeros((0,1))
    #
    stored_exit_time = np.zeros(config['exit_trial_num'])
    for sharpness in sharpness_arr:
        sample_id = rank
        while sample_id < config['exit_trial_num']:
            stored_exit_time[sample_id] = get_exit_time(config,
                                                        config['lr_min'],
                                                        sharpness,
                                                        config['batch_size_min'],
                                                        config['r_min'])
            sample_id += comm_size
        comm.Barrier()
        if rank==0:
            reduced_exit_time = np.zeros_like(stored_exit_time) 
        else:
            reduced_exit_time = None 
        comm.Reduce( [stored_exit_time, MPI.DOUBLE],
                     [reduced_exit_time, MPI.DOUBLE],
                     op = MPI.SUM, root = 0)
        if rank==0:
            exit_time_arr = np.append(exit_time_arr, np.mean(reduced_exit_time))
            std_arr = np.append(std_arr, np.std(reduced_exit_time))
            log_exit_time_arr = np.append(log_exit_time_arr, np.mean(np.std(np.log(reduced_exit_time))))
            log_std_arr = np.append(log_std_arr, np.std(np.log(reduced_exit_time)))
    return (sharpness_arr, exit_time_arr, std_arr, log_exit_time_arr, log_std_arr)

def get_batch_size_vs_exit_time(config, comm):
    comm_size = comm.Get_size()
    rank = comm.Get_rank()
    bs_arr = np.linspace(config['batch_size_min'], 
                         config['batch_size_min'] + config['batch_size_interval'] - 1,
                         config['interval_sample']).astype(int)
    exit_time_arr = np.zeros((0,1))
    log_exit_time_arr = np.zeros((0,1))
    std_arr = np.zeros((0,1))
    log_std_arr = np.zeros((0,1))
    #
    stored_exit_time = np.zeros(config['exit_trial_num'])
    for bs in bs_arr:
        sample_id = rank
        while sample_id < config['exit_trial_num']:
            stored_exit_time[sample_id] = get_exit_time(config,
                                                        config['lr_min'],
                                                        config["sharpness_min"],
                                                        bs,
                                                        config['r_min'])
            sample_id += comm_size
        comm.Barrier()
        if rank==0:
            reduced_exit_time = np.zeros_like(stored_exit_time) 
        else:
            reduced_exit_time = None 
        comm.Reduce( [stored_exit_time, MPI.DOUBLE],
                     [reduced_exit_time, MPI.DOUBLE],
                     op = MPI.SUM, root = 0)
        if rank==0:
            exit_time_arr = np.append(exit_time_arr, np.mean(reduced_exit_time))
            std_arr = np.append(std_arr, np.std(reduced_exit_time))
            log_exit_time_arr = np.append(log_exit_time_arr, np.mean(np.std(np.log(reduced_exit_time))))
            log_std_arr = np.append(log_std_arr, np.std(np.log(reduced_exit_time)))
    return (bs_arr, exit_time_arr, std_arr, log_exit_time_arr, log_std_arr)

def get_r_vs_exit_time(config, comm):
    comm_size = comm.Get_size()
    rank = comm.Get_rank()
    r_arr = np.linspace(config['r_min'], 
                         config['r_min'] + config['r_interval'],
                         config['interval_sample'])
    exit_time_arr = np.zeros((0,1))
    log_exit_time_arr = np.zeros((0,1))
    std_arr = np.zeros((0,1))
    log_std_arr = np.zeros((0,1))
    #
    stored_exit_time = np.zeros(config['exit_trial_num'])
    for r in r_arr:
        sample_id = rank
        while sample_id < config['exit_trial_num']:
            stored_exit_time[sample_id] = get_exit_time(config,
                                                        config['lr_min'],
                                                        config["sharpness_min"],
                                                        config["batch_size_min"],
                                                        r)
            sample_id += comm_size
        comm.Barrier()
        if rank==0:
            reduced_exit_time = np.zeros_like(stored_exit_time) 
        else:
            reduced_exit_time = None 
        comm.Reduce( [stored_exit_time, MPI.DOUBLE],
                     [reduced_exit_time, MPI.DOUBLE],
                     op = MPI.SUM, root = 0)
        if rank==0:
            exit_time_arr = np.append(exit_time_arr, np.mean(reduced_exit_time))
            std_arr = np.append(std_arr, np.std(reduced_exit_time))
            log_exit_time_arr = np.append(log_exit_time_arr, np.mean(np.std(np.log(reduced_exit_time))))
            log_std_arr = np.append(log_std_arr, np.std(np.log(reduced_exit_time)))
    return (r_arr, exit_time_arr, std_arr, log_exit_time_arr, log_std_arr)

import matplotlib.pyplot as plt
from scipy import stats
def draw(config_fn, sharpness_results, lr_results, batch_size_results, r_results):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 3, figsize=(12, 16))
    plt.suptitle(config_fn, x=0.05, y=1)
    # Sharpness 
    (x, y, std, log_std, _) = sharpness_results
    y += 0.00001
    coeff, _ = stats.pearsonr(x, y)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    #
    ax1[0].set_xlabel("sharpness")
    ax1[0].set_ylabel("exit time")
    ax1[0].errorbar(x, y, yerr=std, fmt='.k') 
    ax1[0].plot(x, m*x + c) 
    ax1[0].set_ylim(bottom=0, top=None)
    ax1[0].legend([f'Corr: {coeff:.3g}'])
    # Log
    x_2 = x
    y_2 = np.log(y)
    coeff_2, _ = stats.pearsonr(x_2, y_2)
    A = np.vstack([x_2, np.ones(len(x))]).T
    m_2, c_2 = np.linalg.lstsq(A, y_2, rcond=None)[0]
    ax1[1].set_xlabel("sharpness")
    ax1[1].set_ylabel("log(exit time)")
    print(y_2.shape)
    print(log_std.shape)
    ax1[1].errorbar(x_2, y_2, yerr=log_std, fmt='.k') 
    ax1[1].plot(x_2, m_2*x_2 + c_2)
    ax1[1].set_ylim(bottom=0, top=None)
    ax1[1].legend([f'Corr: {coeff_2:.3g}'])
    ax1[1].set_title(f'tau = exp(sharpness)')
    #
    x_3 = x
    y_3 = np.log(y)**2
    coeff_3, _ = stats.pearsonr(x_3, y_3)
    A = np.vstack([x_3, np.ones(len(x_3))]).T
    m_3, c_3 = np.linalg.lstsq(A, y_3, rcond=None)[0]
    ax1[2].set_xlabel("sharpness")
    ax1[2].set_ylabel("log(exit time)^2")
    ax1[2].errorbar(x_3, y_3, yerr=std*0, fmt='.k') 
    ax1[2].plot(x_3, m_3*x_3 + c_3)
    ax1[2].set_ylim(bottom=0, top=None)
    ax1[2].legend([f'Corr: {coeff_3:.3g}'])
    ax1[2].set_title(f'tau = exp(sharpness^(1/2))')
    ###
    # Learning rate
    (x, y, std, log_std, _) = lr_results
    y += 0.00001
    #
    coeff, _ = stats.pearsonr(x, y)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    ax2[0].set_xlabel("lr")
    ax2[0].set_ylabel("exit time")
    ax2[0].errorbar(x, y, yerr=std, fmt='.k') 
    ax2[0].plot(x, m*x + c) 
    ax2[0].set_ylim(bottom=0, top=None)
    ax2[0].legend([f'Corr: {coeff:.3g}'])
    # Log
    x_2 = x
    y_2 = np.log(y)
    coeff_2, _ = stats.pearsonr(x_2, y_2)
    A = np.vstack([x_2, np.ones(len(x))]).T
    m_2, c_2 = np.linalg.lstsq(A, y_2, rcond=None)[0]
    ax2[1].set_xlabel("lr")
    ax2[1].set_ylabel("log(exit time)")
    ax2[1].errorbar(x_2, y_2, yerr=log_std, fmt='.k') 
    ax2[1].plot(x_2, m_2*x_2 + c_2)
    ax2[1].set_ylim(bottom=0, top=None)
    ax2[1].legend([f'Corr: {coeff_2:.3g}'])
    ax2[1].set_title(f'tau = exp(lr^(-1))')
    #
    x_3 = x
    y_3 = np.log(y)**2
    coeff_3, _ = stats.pearsonr(x_3, y_3)
    A = np.vstack([x_3, np.ones(len(x_3))]).T
    m_3, c_3 = np.linalg.lstsq(A, y_3, rcond=None)[0]
    ax2[2].set_xlabel("lr")
    ax2[2].set_ylabel("log(exit time)^2")
    ax2[2].errorbar(x_3, y_3, yerr=std, fmt='.k') 
    ax2[2].plot(x_3, m_3*x_3 + c_3)
    ax2[2].set_ylim(bottom=0, top=None)
    ax2[2].legend([f'Corr: {coeff_3:.3g}'])
    ax2[2].set_title(f'tau = exp(lr^(-1/2))')
    # Batch size
    (x, y, std, log_std, _) = batch_size_results
    y += 0.00001
    #
    coeff, _ = stats.pearsonr(x, y)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    ax3[0].set_xlabel("batch_size")
    ax3[0].set_ylabel("exit time")
    ax3[0].errorbar(x, y, yerr=std, fmt='.k') 
    ax3[0].plot(x, m*x + c) 
    ax3[0].set_ylim(bottom=0, top=None)
    ax3[0].legend([f'Corr: {coeff:.3g}'])
    # Log
    x_2 = x
    y_2 = np.log(y)
    coeff_2, _ = stats.pearsonr(x_2, y_2)
    A = np.vstack([x_2, np.ones(len(x))]).T
    m_2, c_2 = np.linalg.lstsq(A, y_2, rcond=None)[0]
    ax3[1].set_xlabel("batch size")
    ax3[1].set_ylabel("log(exit time)")
    ax3[1].errorbar(x_2, y_2, yerr=log_std, fmt='.k') 
    ax3[1].plot(x_2, m_2*x_2 + c_2)
    ax3[1].set_ylim(bottom=0, top=None)
    ax3[1].legend([f'Corr: {coeff_2:.3g}'])
    ax3[1].set_title(f'tau = exp(batch size)')
    # R
    (x, y, std, log_std, _) = r_results
    y += 0.00001
    #
    coeff, _ = stats.pearsonr(x, y)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    ax4[0].set_xlabel("r")
    ax4[0].set_ylabel("exit time")
    ax4[0].errorbar(x, y, yerr=std, fmt='.k') 
    ax4[0].plot(x, m*x + c) 
    ax4[0].set_ylim(bottom=0, top=None)
    ax4[0].legend([f'Corr: {coeff:.3g}'])
    # Log
    x_2 = x
    y_2 = np.log(y)
    coeff_2, _ = stats.pearsonr(x_2, y_2)
    A = np.vstack([x_2, np.ones(len(x))]).T
    m_2, c_2 = np.linalg.lstsq(A, y_2, rcond=None)[0]
    ax4[1].set_xlabel("r")
    ax4[1].set_ylabel("log(exit time)")
    ax4[1].errorbar(x_2, y_2, yerr=log_std, fmt='.k') 
    ax4[1].plot(x_2, m_2*x_2 + c_2)
    ax4[1].set_ylim(bottom=0, top=None)
    ax4[1].legend([f'Corr: {coeff_2:.3g}'])
    ax4[1].set_title(f'tau = exp(r)')
    # Log
    x_3 = x
    y_3 = np.sqrt(np.log(y))
    coeff_3, _ = stats.pearsonr(x_3, y_3)
    A = np.vstack([x_3, np.ones(len(x))]).T
    m_3, c_3 = np.linalg.lstsq(A, y_3, rcond=None)[0]
    ax4[2].set_xlabel("r")
    ax4[2].set_ylabel("sqrt(log(exit time))")
    ax4[2].errorbar(x_3, y_3, yerr=std*0, fmt='.k') 
    ax4[2].plot(x_3, m_3*x_3 + c_3)
    ax4[2].set_ylim(bottom=0, top=None)
    ax4[2].legend([f'Corr: {coeff_3:.3g}'])
    ax4[2].set_title(f'tau = exp(r^2)')

    plt.tight_layout()
    plt.show()
    fig.savefig("results.png",dpi=300)

import os
import time
import numpy as np
import sys
import optuna
from mpi4py import MPI
import json
def main():
    config_fn = 'MLP_SGD.json' 
    with open(config_fn) as json_file:
        config = json.load(json_file)
    #
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    rank = comm.Get_rank()

    if len(sys.argv) > 1 and sys.argv[1] == "sanity_check":
        config['exit_trial_num'] = 10 
        config['interval_sample'] = 2 
        if rank == 0:
            report("Runnig Sanity Check")
            print(end="", flush=True)
    set_device(config)
    if rank == 0:
        if 0<=config["core"]<torch.cuda.device_count() and torch.cuda.is_available():
            report(f'use GPU; core:{config["core"]}')
        else:
            report('use CPU in this trial')
        print(end="", flush=True)
    #
    global dataset 
    dataset = get_dataset(config)
    if rank == 0 and config["model"] == 'MLP':
        report("Training Started")
        print( end="", flush=True)
        model = get_model(config, 1)
        for _ in range(10000):
            model(dataset.train.x).backward()
            model.update(0.0001)
        torch.save(model.state_dict(), "./MLP_init_params.pt")
    comm.Barrier()
    #
    # Sharpess
    if rank == 0:
        report("Sharpness Analysis Started")
        print( end="", flush=True)
    sharpness_results = get_sharpness_vs_exit_time(config, comm)
    # Learning ratae
    if rank == 0:
        report("LR Analysis Started")
        print(end="", flush=True)
    lr_results         = get_lr_vs_exit_time(config, comm)
    # batchsize
    if rank == 0:
        report("Batch Sizes Analysis Started")
        print(end="", flush=True)
    batch_size_results = get_batch_size_vs_exit_time(config, comm)
    # r
    if rank == 0:
        report("r Analysis Started")
        print(end="", flush=True)
    r_results = get_r_vs_exit_time(config, comm)
    if rank == 0:
        report("Drawing Started")
        print(end="", flush=True)
    if rank == 0: draw(config_fn, sharpness_results, lr_results, batch_size_results, r_results)
    if rank == 0:
        report("Done")
        print(end="", flush=True)
    

if __name__=='__main__':
    main()