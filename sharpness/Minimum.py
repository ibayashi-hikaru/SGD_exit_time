#!/usr/bin/env python
# coding: utf-8
# -------------------------------------------------------------------
import datetime
def report(*args):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+' '+' '.join(map(str,args)).replace('\n',''))


# -------------------------------------------------------------------
import torch
class _LossFunctionOverAlpha(torch.nn.Module):
    def __init__(self, weights, target_loss=0):
        super().__init__()
        num_of_layer = len(weights)
        self.exponent = torch.nn.Parameter(torch.zeros(num_of_layer-1).double())
        self.L = num_of_layer-1
        self.target_loss = target_loss

    def forward(self, weights, biases):
        loss = 0
        # alpha^2
        alpha2 = torch.exp(self.exponent)
        prod_scale = 1
        for l in range(self.L):
            loss += weights[l]/alpha2[l]
            prod_scale = prod_scale*alpha2[l]
            loss += biases[l]/prod_scale
        loss += weights[self.L]*prod_scale
        loss += biases[self.L]
        return torch.abs(loss - self.target_loss) + self.target_loss

    def get_alpha(self):
        return torch.exp(self.exponent/2)

# -------------------------------------------------------------------
import math
def _get_total_sharpness(weights,biases,target_sharpness=0,lr=1e-1,num_epoch=1000,decay=1e-3):
    # normalize 
    scale = 0
    scale = max(scale,max(weights.values()))
    scale = max(scale,max(biases.values()))
    for k in weights:   weights[k] /= scale
    for k in biases:    biases[k]  /= scale

    while True:
        # record
        trails = {'loss':[], 'scale':scale, 'lr':lr}

        # get ready optimizer and scheduler 
        model = _LossFunctionOverAlpha(weights, target_sharpness/scale)
        model = model.double()
        optimizer = torch.optim.SGD(model.parameters(), lr)     # best lr=1e-0, faster than Adam? 
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda e: 1/(e*decay+1))

        # optimize
        for epoch in range(num_epoch):
            loss = model(weights, biases)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss = (loss*scale).item()
            trails['loss'].append(loss)
            if math.isnan(loss):
                print('nan found, use smaller steps-size', lr)
                del trails, model, optimizer, scheduler
                lr /= 2
                break
        else:
            return loss, model.get_alpha() 


# -------------------------------------------------------------------
from .tools import Hessian_trace, Hessian_diag 
def sharpness(model, train_x, diagH):
    weights, biases = {},{}
    for name,param in diagH.named_parameters():
        if not name.startswith('layer'): continue
        name = name.replace('layer','')

        if name.endswith('.weight'):    # layer[i].weight
            lth             = int( name.replace('.weight','') )
            weights[lth]    = param.double().sum().item()
        if name.endswith('.bias'):      # layer[i].bias
            lth             = int( name.replace('.bias','') )
            biases[lth]     = param.double().sum().item()
    minimum_sharpness, _ = _get_total_sharpness(weights, biases)
    return minimum_sharpness

def update_param(model, train_x, diagH, target_sharpness=0):
    weights, biases = {},{}
    for name,param in diagH.named_parameters():
        if not name.startswith('layer'): continue
        name = name.replace('layer','')

        if name.endswith('.weight'):    # layer[i].weight
            lth             = int( name.replace('.weight','') )
            weights[lth]    = param.double().sum().item()
        if name.endswith('.bias'):      # layer[i].bias
            lth             = int( name.replace('.bias','') )
            biases[lth]     = param.double().sum().item()
    _, alpha = _get_total_sharpness(weights, biases, target_sharpness)
    #
    num_of_layer = len(weights)
    L = num_of_layer-1
    prod_scale = 1
    for name, param in model.named_parameters():
        if not name.startswith('layer'): continue
        name = name.replace('layer','')

        if name.endswith('.weight'):    # layer[i].weight
            lth = int( name.replace('.weight','') )
            if lth != L:
                param.data *= alpha[lth]
                prod_scale = prod_scale*alpha[lth]
            else:
                param.data /= prod_scale
        if name.endswith('.bias'):      # layer[i].bias
            lth = int( name.replace('.bias','') )
            if lth != L:
                param.data *= prod_scale
            else:
                continue
#