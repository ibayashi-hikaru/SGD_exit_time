from utility import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float
#
dataset = None
#
import torch
import copy
def get_end_point(config):
    model = get_model(config)
    model.init_params()
    exit_time = 0
    trajectory = np.zeros((config["num_itr"], config["num_dim"]))
    for itr in range(config["num_itr"]):
        model().backward()
        model.update(config["lr"])
        model.perturb(config["lr"])
        trajectory[itr] = model.layer0.weight.detach().cpu().numpy()
    return trajectory

def get_distribution(config, comm):
    comm_size = comm.Get_size()
    rank = comm.Get_rank()
    data = np.zeros((config['num_sample'], config["num_itr"], config["num_dim"]))
    sample_id = rank
    while sample_id < config['num_sample']:
        data[sample_id] = get_end_point(config)
        sample_id += comm_size
    comm.Barrier()
    if rank==0:
        total = np.zeros_like(data) 
    else:
        total = None 
    comm.Reduce( [data, MPI.DOUBLE], [total, MPI.DOUBLE], op = MPI.SUM, root = 0)
    if rank==0:
        return total

import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
def draw(config, distribution):
    n_bins = 20
    model = get_model(config)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True, tight_layout=True)
    ax1.hist(distribution, bins=n_bins, density=True)
    #
    x = np.linspace(-0.002, 0.002, 50)
    y = x**2
    ax2.plot(x, y)
    # ax2.set_ylim([0, 0.000004])
    plt.suptitle(datetime.datetime.now().strftime('%H:%M:%S'))
    plt.show()
    fig.savefig("results.png", dpi=300)

import os
import time
import numpy as np
import sys
import optuna
from mpi4py import MPI
import ot
def main():
    config = {}
    config['core'] = 0 
    config['seed'] = -1
    config['dims'] = np.arange(100) + 1
    config["num_itr"] = 1000
    config["num_sample"] = 100
    config["lr"] = 0.01
    config["threshold"] = 0.01
    #
    # config['model'] = 'quad_func'
    config["model"] = 'styblinski_tang_func'

    #
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    rank = comm.Get_rank()

    if len(sys.argv) > 1 and sys.argv[1] == "sanity_check":
        config['num_sample'] = 10 
        if rank == 0:
            report("Runnig Sanity Check")
            print(end="", flush=True)
    set_device(config, rank)

    if rank == 0:
        report("Process Started")
        print(end="", flush=True)
    distribution_dict = {}
    for dim in config["dims"]:
        if rank == 0:
            report(f"{dim}")
            print(end="", flush=True)
        config["num_dim"] = dim
        distribution_dict[dim] = get_distribution(config, comm)
    if rank == 0:
        # draw(config, distribution)
        for dim in config["dims"]:
            distribution_dict[dim] = distribution_dict[dim].transpose((1,0,2))
        a, b = np.ones((config["num_sample"],)) / config["num_sample"], np.ones((config["num_sample"],)) / config["num_sample"]  # uniform distribution on samples
        itr_num = []
        for dim in config["dims"]:
            init_dist = np.zeros((config["num_sample"], dim))
            report(f"{dim}")
            print(end="", flush=True)
            needed_itr = -1
            for itr in range(config["num_itr"]):
                # loss matrix
                if itr == 0:
                    M = ot.dist(init_dist,
                                distribution_dict[dim][itr])
                else:
                    M = ot.dist(distribution_dict[dim][itr - 1],
                                distribution_dict[dim][itr])
                M /= M.max()
                distance = ot.emd2(a, b, M)
                if distance < config['threshold']:
                    needed_itr = itr
                    break
            itr_num.append(needed_itr)

        font = {'family' : 'normal',
                'size'   : 15}
        
        matplotlib.rc('font', **font)
        x = config["dims"]
        print(len(itr_num))
        y = itr_num

        plt.figure()
        plt.title('Convergence to Stationary Distribution')
        plt.plot(x, y)
        plt.xlabel('Dimension')
        plt.ylabel('Number of Iteraitons')
        plt.tight_layout()
        legend = []
        plt.savefig("Fig1.pdf")

        report("Process Ended")
        print(end="", flush=True)


if __name__=='__main__':
    main()