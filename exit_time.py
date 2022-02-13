from utility import *
import argparse
args = argparse.ArgumentParser()
args.add_argument('--setup', default='', type=str)
args.add_argument('--sanity_check', action='store_true')
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
        model.zero_grad()
        if config["optim"] == "SGD":
            data_size = dataset.train.x.size()[0]
            shuffled_data = dataset.train.x[torch.randperm(data_size)]
            mini_batch = shuffled_data[:batch_size,:]
            model(mini_batch).backward()
            model.update(lr)
        elif config["optim"] == "SGLD":
            # Full gradient
            split_num = 1000
            for batch in torch.split(dataset.train.x, split_num):  
                model(batch).backward()
            model.update(lr / split_num)
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


import os
import time
import numpy as np
import sys
from mpi4py import MPI
import json
def main():
    setup = args.parse_args().setup
    config_fn = setup + "/config.json"
    with open(config_fn) as json_file:
        config = json.load(json_file)
    #
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    rank = comm.Get_rank()

    if args.parse_args().sanity_check:
        config['exit_tril_num'] = 10 
        config['interval_sample'] = 2 
        report(rank, "Runnig Sanity Check")
    #
    set_device(config)
    if 0<=config["core"]<torch.cuda.device_count() and torch.cuda.is_available():
        report(rank, f'use GPU; core:{config["core"]}')
    else:
        report(rank, 'use CPU in this trial')
    #
    global dataset 
    dataset = get_dataset(config)
    # if rank == 0 and config["model"] == 'MLP':
    #     report(rank, "Training Started")
    #     print( end="", flush=True)
    #     model = get_model(config, 1)
    #     for itr in range(10000):
    #         if itr % 100 == 0: report(rank, f"{itr}/10000")
    #         split_num = 1000 
    #         for batch in torch.split(dataset.train.x, split_num):  
    #             model(batch).backward()
    #         model.update(0.0001/split_num)
    #     torch.save(model.state_dict(), "./MLP_init_params.pt")
    # comm.Barrier()
    #
    report(rank, "Sharpness Analysis Started")
    sharpness_results = get_sharpness_vs_exit_time(config, comm)
    #
    report(rank, "LR Analysis Started")
    lr_results = get_lr_vs_exit_time(config, comm)
    #
    report(rank, "Batch Sizes Analysis Started")
    batch_size_results = get_batch_size_vs_exit_time(config, comm)
    #
    report(rank, "R Analysis Started")
    r_results = get_r_vs_exit_time(config, comm)
    #
    report(rank, "Saving")
    if rank == 0:
        np.save(f"./{setup}/sharpness_results" , sharpness_results)
        np.save(f"./{setup}/lr_results"        , lr_results)
        np.save(f"./{setup}/batch_size_results", batch_size_results)
        np.save(f"./{setup}/r_results"         , r_results)
    report(rank, "Done")
    

if __name__=='__main__':
    main()