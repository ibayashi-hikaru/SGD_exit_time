import torch
import math
import os
import numpy as np
from utility import *
def optimize(dataset, model, config):
    loss_func = get_loss(config)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], weight_decay=1e-5, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500,1000,2000,3000,4000], gamma=0.5)

    def update():
        model.train()
        measure = {'loss':0,'accuracy':0}
        index = torch.randperm(dataset.train.n)
        for idx in torch.split(index, config["batch_size"]):
            optimizer.zero_grad()
            x = dataset.train.x[idx]
            y = dataset.train.y[idx]
            o = model(x)
            loss = loss_func(o,y)
            loss.backward()            
        optimizer.step()
        scheduler.step()

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
    while True:
        epoch += 1
        report(f'epoch:{epoch}')
        #
        update()
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
            # report(*message)
        assert not math.isnan(status['train']['loss']), 'find nan in train-loss'
        assert not math.isnan(status['test']['loss']),  'find nan in test-loss'
        if epoch == config["epoch"]: break

