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
    args.add_argument('--core',           default=0, type=int)
    args.add_argument('--model',          default='MLP', type=str)
    args.add_argument('--data',           default='CIFAR10', type=str)
    # optim 
    args.add_argument('--batch_size',     default=2**7, type=int)
    args.add_argument('--lr',             default=1e-3, type=float)
    args.add_argument('--threshold',      default=1, type=float)
    args.add_argument('--loss', default="MSE", type=str)
    return args.parse_args()

# -------------------------------------------------------------------
import torch
import math
from sharpness.tools import Hessian_trace, Hessian_diag 
import sharpness.Minimum as Minimum
import os
def optimize(dataset, model, config, out_dir):
    loss_func = get_loss(config)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], weight_decay=1e-5, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 150], gamma=0.5)

    def ERM_update():
        model.train()
        measure = {'loss':0,'accuracy':0}
        index = torch.randperm(dataset.train.n)
        for idx in torch.split(index, config["batch_size"]):
            x = dataset.train.x[idx]
            y = dataset.train.y[idx]
            o = model(x)
            loss = loss_func(o,y)
            optimizer.zero_grad()
            loss.backward()            
            optimizer.step()
            scheduler.step()

    def sharpness_update():
        batch_size = 256 
        diagH = Hessian_diag.effective(dataset.train.x, model, batch_size)
        Minimum.update_param(model, dataset.train.x, diagH)

    def evaluate(data):
        model.eval()
        with torch.no_grad():
            output    = model(data.x)            # logit
            loss    = loss_func(output, data.y)
            loss     = loss.item()
            output    = output.max(dim=1)[1]            # logit -> index
            correct    = (output==data.y) 
            accuracy   = correct.float().mean().item()    # bool -> int(0,1) -> float
            return loss, accuracy

    epoch = 0
    itr = 0
    while True:
        epoch += 1
        report(f'epoch:{epoch}')
        #
        ERM_update()
        #
        status = {}
        train_loss, train_acc = evaluate(dataset.train)
        status['train'] = {'loss': train_loss, 'accuracy':  train_acc}
        test_loss, test_acc = evaluate(dataset.test)
        status['test'] = {'loss': test_loss, 'accuracy':  test_acc}
        # Dump Log
        for mode in ['train','test']:
            message     = [f'\t{mode:5}']
            message    += [f"loss:{status[mode]['loss']: 18.7f}"]
            message    += [f"accuracy:{status[mode]['accuracy']: 9.7f}"]
            # Save Weight
            itr += 1
            torch.save(model.state_dict(), out_dir + f"/{itr:05}.weight")
            report(*message)
        assert not math.isnan(status['train']['loss']), 'find nan in train-loss'
        assert not math.isnan(status['test']['loss']),  'find nan in test-loss'
        # Save Weight
        # torch.save(model.state_dict(), out_dir + f"/{epoch:05}.weight")
        if train_loss <= config["threshold"]:
            report(f"Saving weight at epoch {epoch}")
            break


# -------------------------------------------------------------------    
import os
import time
import json
import shutil
import random
def main():
    config_fn = "./config.json"
    with open(config_fn) as json_file:
        config = json.load(json_file)
    #
    set_device(config)
    #
    dataset = get_dataset(config)
    #
    for i in range(1):
        report(f"Trial {i}")
        seed = int(time.time())
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)    
        model = get_model(config)
        os.system(f"rm -r results/trial_{i}")
        os.makedirs(f"results/trial_{i}", exist_ok=True)
        optimize(dataset, model, config, f"results/trial_{i}")

if __name__=='__main__':
    main()

# -------------------------------------------------------------------

