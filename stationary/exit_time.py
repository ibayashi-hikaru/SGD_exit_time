from utility import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float
#
dataset = None
#
import torch
import copy
def get_exit_time(config, lr, sharpness, batch_size):
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
    while(distance(init_model, model) < config['r']):
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
    data = np.zeros(config['exit_trial_num'])
    exit_time_arr = np.zeros((0,1))
    std_arr = np.zeros((0,1))
    for lr in lr_arr:
        sample_id = rank
        while sample_id < config['exit_trial_num']:
            data[sample_id] = get_exit_time(config,
                                            lr,
                                            config['sharpness_min'],
                                            config['batch_size_min'])
            sample_id += comm_size
        comm.Barrier()
        if rank==0:
            total = np.zeros_like(data) 
        else:
            total = None 
        comm.Reduce( [data, MPI.DOUBLE], [total, MPI.DOUBLE], op = MPI.SUM, root = 0)
        if rank==0:
            exit_time_arr = np.append(exit_time_arr, np.mean(total))
            std_arr = np.append(std_arr, np.std(total))
    return (lr_arr, exit_time_arr, std_arr)

def get_sharpness_vs_exit_time(config, comm):
    comm_size = comm.Get_size()
    rank = comm.Get_rank()
    sharpness_arr = np.linspace(config['sharpness_min'], 
                                config['sharpness_min']+config['sharpness_interval'],
                                config['interval_sample'])
    data = np.zeros(config['exit_trial_num'])
    exit_time_arr = np.zeros((0,1))
    std_arr = np.zeros((0,1))
    for sharpness in sharpness_arr:
        sample_id = rank
        while sample_id < config['exit_trial_num']:
            data[sample_id] = get_exit_time(config,
                                            config['lr_min'],
                                            sharpness,
                                            config['batch_size_min'])
            sample_id += comm_size
        comm.Barrier()
        if rank==0:
            total = np.zeros_like(data) 
        else:
            total = None 
        comm.Reduce( [data, MPI.DOUBLE], [total, MPI.DOUBLE], op = MPI.SUM, root = 0)
        if rank==0:
            exit_time_arr = np.append(exit_time_arr, np.mean(total))
            std_arr = np.append(std_arr, np.std(total))
    return (sharpness_arr, exit_time_arr, std_arr)

def get_batch_size_vs_exit_time(config, comm):
    comm_size = comm.Get_size()
    rank = comm.Get_rank()
    bs_arr = np.linspace(config['batch_size_min'], 
                         config['batch_size_min'] + config['batch_size_interval'] - 1,
                         config['interval_sample']).astype(int)
    data = np.zeros(config['exit_trial_num'])
    exit_time_arr = np.zeros((0,1))
    std_arr = np.zeros((0,1))
    for bs in bs_arr:
        sample_id = rank
        while sample_id < config['exit_trial_num']:
            data[sample_id] = get_exit_time(config,
                                            config['lr_min'],
                                            config["sharpness_min"],
                                            bs)
            sample_id += comm_size
        comm.Barrier()
        if rank==0:
            total = np.zeros_like(data) 
        else:
            total = None 
        comm.Reduce( [data, MPI.DOUBLE], [total, MPI.DOUBLE], op = MPI.SUM, root = 0)
        if rank==0:
            exit_time_arr = np.append(exit_time_arr, np.mean(total))
            std_arr = np.append(std_arr, np.std(total))
    return (bs_arr, exit_time_arr, std_arr)
import matplotlib.pyplot as plt
from scipy import stats
def draw(sharpness_results, lr_results, batch_size_results):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 2, figsize=(9, 12))
    def draw_subfig(ax, x, y, std, h_param_name):
        coeff, _ = stats.pearsonr(x, y)
        log_coeff, _ = stats.pearsonr(x,np.log(y))
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        log_m, log_c = np.linalg.lstsq(A, np.log(y), rcond=None)[0]
        #
        ax[0].set_xlabel(h_param_name)
        ax[0].set_ylabel("exit time")
        ax[0].errorbar(x, y, yerr=std, fmt='.k') 
        ax[0].plot(x, m*x + c) 
        ax[0].set_ylim(bottom=0, top=None)
        ax[0].legend([f'Corr: {coeff:.3g}'])

        # Log
        ax[1].set_xlabel(h_param_name)
        ax[1].set_ylabel("log(exit time)")
        ax[1].errorbar(x, y, yerr=std, fmt='.k') 
        ax[1].plot(x, np.exp(log_m*x + log_c)) 
        ax[1].set_yscale("log") 
        ax[1].legend([f'Corr: {log_coeff:.3g}'])
    # Sharpness 
    draw_subfig(ax1, *sharpness_results, "sharpness")
    # Learning rate
    draw_subfig(ax2, *lr_results, "lr")
    # Learning rate
    draw_subfig(ax3, *batch_size_results, "batch_size")

    plt.suptitle(datetime.datetime.now().strftime('%H:%M:%S'))
    plt.tight_layout()
    plt.show()
    fig.savefig("results.png",dpi=300)

import os
import time
import numpy as np
import sys
import optuna
from mpi4py import MPI
def main():
    config = {}
    config['core'] = 0 
    config['seed'] = -1
    config['num_dim'] = 100
    config['sharpness_min'] = 1
    config['sharpness_interval'] = 10 
    config['r'] = 0.8 
    config['lr_min'] = 0.001
    config['lr_interval'] = 0.005 
    config['batch_size_min'] = 2 
    config['batch_size_interval'] = 10 
    config['exit_trial_num'] = 100 
    config['interval_sample'] = 10 
    config['optim'] = "SGLD"
    #
    config['data'] = 'AVILA2'
    # config['model'] = 'quad_func'
    # config['model'] = 'styblinski_tang_func'
    config['model']   = 'MLP'

    #
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    rank = comm.Get_rank()

    if sys.argv[1] == "sanity_check":
        config['exit_trial_num'] = 10 
        config['interval_sample'] = 2 
        if rank == 0:
            report("Runnig Sanity Check")
            print(end="", flush=True)
    set_device(config)
    #
    global dataset 
    dataset = get_dataset(config)
    if rank == 0 and config["model"] == 'MLP':
        for _ in range(100):
            model = get_model(config, 1)
            model(dataset.train.x).backward()
            model.update(0.001)
            torch.save(model.state_dict(), "./MLP_init_params.pt")
    #
    # Sharpess
    if rank == 0:
        report("Sharpness Analysis Started")
        print( end="", flush=True)
    sharpness_results  = get_sharpness_vs_exit_time(config, comm)
    # Learning ratae
    if rank == 0:
        report("LR Analysis Started")
        print(end="", flush=True)
    lr_results         = get_lr_vs_exit_time(config, comm)
    # batchsize
    if rank == 0:
        report("Batch Sizse Analysis Started")
        print(end="", flush=True)
    batch_size_results = get_batch_size_vs_exit_time(config, comm)
    if rank == 0: draw(sharpness_results, lr_results, batch_size_results)
    

if __name__=='__main__':
    main()