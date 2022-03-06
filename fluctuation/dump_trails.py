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
args = argparse.ArgumentParser()
args.add_argument('--config_fn', default='', type=str)


# -------------------------------------------------------------------
import torch
import os, fnmatch
import copy
import pickle
from hessian_eigenthings import compute_hessian_eigenthings
def dump_trails(dataset, model, config, trial_dir):
    def train_loss():
        model.eval()
        with torch.no_grad():
            output    = model(dataset.train.x)            # logit
            loss    = loss_func(output, dataset.train.y)
            return loss.item()

    def train_accuracy():
        model.eval()
        with torch.no_grad():
            output    = model(dataset.train.x)            # logit
            output    = output.max(dim=1)[1]
            correct    = (output==dataset.train.y)        # index -> bool
            return  correct.float().mean().item()    # bool -> int(0,1) -> float

    def test_loss():
        model.eval()
        with torch.no_grad():
            output    = model(dataset.test.x)            # logit
            loss    = loss_func(output, dataset.test.y)
            return loss.item()

    def test_accuracy():
        model.eval()
        with torch.no_grad():
            output    = model(dataset.test.x)            # logit
            output    = output.max(dim=1)[1]
            correct    = (output==dataset.test.y)        # index -> bool
            return  correct.float().mean().item()    # bool -> int(0,1) -> float


    def l2_norm():
        norm = 0
        for param in model.parameters():
            norm += torch.norm(param.data, p=2).item()
        return  norm

    def eigen_sharpness():
        num_eigenthings = 1 
        max_eigenval, _ = compute_hessian_eigenthings(model, train_dataloader, loss_func, num_eigenthings)
        return max_eigenval[0]

    def keskar_sharpness(): 
        origin = copy.deepcopy(model)
        maxima = copy.deepcopy(model)
        
        keskar_optimizer = torch.optim.SGD(maxima.parameters(),lr=0.01)
        keskar_scheduler = torch.optim.lr_scheduler.LambdaLR(keskar_optimizer, lr_lambda = lambda e: 1)
        
        def get_maxima():
            maxima.eval()
            with torch.no_grad():
                output = maxima(dataset.train.x)
                loss = loss_func(output, dataset.train.y)
                return loss.item()
    
        fx = get_maxima()
        worst_loss = fx
        for _ in range(5):
            maxima.train()
            index = torch.randperm(dataset.train.n)
            batch_size = 5000
            for idx in torch.split(index, batch_size):
                keskar_optimizer.zero_grad()
                loss = -loss_func(maxima(dataset.train.x[idx]), dataset.train.y[idx])
                loss.backward()
                keskar_optimizer.step()
                epsilon = 1e-3 
                maxima.clip(origin, epsilon)
                keskar_scheduler.step()
            current_loss = get_maxima()
            if current_loss > worst_loss: worst_loss = current_loss
        
        return (worst_loss-fx)/(1+fx)

    loss_func = get_loss(config)
    train_dataloader, test_dataloader = get_dataloader(config)
    #
    weight_files = fnmatch.filter(os.listdir(trial_dir), '*.weight')
    snapshots = list(map(lambda x: int(x[:5]), weight_files))
    snapshots.sort()
    #
    measure_names = ['train_loss', 'test_loss',\
                     'train_accuracy', 'test_accuracy',\
                     'l2_norm', \
                     'sharpness']
    measure_funcs = [train_loss, test_loss,\
                     train_accuracy, test_accuracy,\
                     l2_norm,\
                     keskar_sharpness]
    for snapshot_id in snapshots:
        report(f"Snapshot at step {snapshot_id}")
        status = {}
        status['t'] = snapshot_id
        model.load_state_dict(torch.load(trial_dir + f"/{snapshot_id:05}.weight"))
        #
        status_fn = trial_dir + f"/{snapshot_id:05}.status"
        if os.path.exists(status_fn):
            with open(status_fn, 'rb') as handle:
                saved_status = pickle.load(handle)
        else:
            saved_status = {}
        for measure_name, measure_func in zip(measure_names, measure_funcs):
            if measure_name in saved_status:
                status[measure_name] = saved_status[measure_name]
                report(f"{measure_name}: {status[measure_name]}")
            else:
                status[measure_name] = measure_func()
                report(f"{measure_name}: {status[measure_name]}")
        with open(trial_dir + f"/{snapshot_id:05}.status", 'wb') as handle:
            pickle.dump(status, handle, protocol=pickle.HIGHEST_PROTOCOL)
# -------------------------------------------------------------------
import os
import json
import pickle
def main():
    config_fn = args.parse_args().config_fn
    with open(config_fn) as json_file:
        config = json.load(json_file)
    #
    set_device(config)
    #
    dataset = get_dataset(config)
    #
    set_device(config)

    dataset = get_dataset(config)
    model = get_model(config)
    #
    for i in range(1):
        report(f"Trial {i}")
        dump_trails(dataset, model, config, f"results/trial_{i}")
    report('finish')    


if __name__=='__main__':
    main()

# -------------------------------------------------------------------

