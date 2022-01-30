#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------------------
import copy
import torch
class BaseNNs(torch.nn.Module):
    def __init__(self, k):
        super(BaseNNs, self).__init__()

    def get_param(self):
        state = copy.deepcopy(self.state_dict())
        for k in state:
            state[k] = state[k].cpu().detach().numpy()
        return state

    def set_param(self, param):
        for k in param:
            param[k] = torch.tensor(param[k],dtype=torch.float)
        self.load_state_dict(param) 

# -------------------------------------------------------------------

class MLP(BaseNNs):
    def __init__(self, k=1.0):
        super(MLP, self).__init__(k)
        self.k = torch.tensor(k) 
        self.k.requires_grad = False
        self.layer0 = torch.nn.Linear(10, 10, bias=False)
        self.layer0.weight.requires_grad = False
        self.layer1 = torch.nn.Linear(10,  2, bias=False)

    def forward(self, x):
        x = x.flatten(1,-1)
        x = self.layer0(x)
        x = torch.relu(x) 
        x = self.layer1(x) 
        x = x * torch.sqrt(self.k) 
        return x
