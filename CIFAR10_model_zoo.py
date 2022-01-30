#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------------------
import copy
import torch
class BaseNNs(torch.nn.Module):
    def __init__(self, dropout_ratio):
        super(BaseNNs, self).__init__()
        self.Dropout   = torch.nn.Dropout(p=dropout_ratio)
        self.Dropout2d = torch.nn.Dropout2d(p=dropout_ratio)

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
# https://github.com/icpm/pytorch-cifar10/blob/master/models/LeNet.py
class LeNet(BaseNNs):
    def __init__(self,dropout_ratio=0.0):
        super(LeNet, self).__init__(dropout_ratio=0.0)
        self.layer0 = torch.nn.Conv2d(3, 6, kernel_size=5)
        self.layer1 = torch.nn.Conv2d(6, 16, kernel_size=5)
        self.layer2 = torch.nn.Linear(16*5*5, 120)
        self.layer3 = torch.nn.Linear(120, 84)
        self.layer4 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer0(x))
        x = self.Dropout2d(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.relu(self.layer1(x))
        x = self.Dropout2d(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.layer2(x))
        x = self.Dropout(x)
        x = torch.nn.functional.relu(self.layer3(x))
        x = self.layer4(x)
        return x


class MLP(BaseNNs):
    def __init__(self, dropout_ratio=0.0):
        super(MLP, self).__init__(dropout_ratio)
        self.layer0 = torch.nn.Linear(3*32*32, 512)
        self.layer1 = torch.nn.Linear(    512, 128)
        self.layer2 = torch.nn.Linear(    128, 512)
        self.layer3 = torch.nn.Linear(    512,  10)

    def forward(self, x):
        x = x.flatten(1,-1)
        x = torch.relu(self.layer0(x))
        x = torch.relu(self.layer1(x))
        x = self.Dropout(x)
        x = torch.relu(self.layer2(x))
        x = self.Dropout(x)
        return self.layer3(x)

# -------------------------------------------------------------------
