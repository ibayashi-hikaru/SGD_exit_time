#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------------------
import copy
import torch
class BaseNNs(torch.nn.Module):
    def __init__(self, dropout_ratio):
        super(BaseNNs, self).__init__()
        self.Dropout = torch.nn.Dropout(p=dropout_ratio)

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
class LeNet(BaseNNs):
    def __init__(self,dropout_ratio=0.0):
        super(LeNet, self).__init__(dropout_ratio)
        self.layer0 = torch.nn.Conv2d( 1,  6,  5, stride=1, padding=2)
        self.layer1 = torch.nn.Conv2d( 6, 16,  5, stride=1, padding=0)
        self.layer2 = torch.nn.Conv2d(16, 30,  5, stride=1, padding=0)
        self.layer3 = torch.nn.Linear(30*7*7, 84)
        self.layer4 = torch.nn.Linear(    84, 10)

    def forward(self, x):                           # := (b,  1,28,28)  # (batch, channel, width, height)
        x = torch.relu(self.layer0(x))              # -> (b,  6,28,28)  # (28)/1 = 28?
        x = torch.nn.functional.max_pool2d(x, 2, 1) # -> (b,  6,27,27)  # stride=1
        x = torch.relu(self.layer1(x))              # -> (b, 16,23,23)
        x = torch.nn.functional.max_pool2d(x, 2, 2) # -> (b, 16,11,11)
        x = torch.relu(self.layer2(x))              # -> (b,120, 7, 7)
        x = x.flatten(1,-1)                         # -> (b,120*7*7)
        x = self.Dropout(x)
        x = torch.relu(self.layer3(x))              # -> (b,84)
        x = self.Dropout(x)
        return self.layer4(x)                       # -> (b,10)

# https://github.com/pluskid/fitting-random-labels/blob/master/model_mlp.py
# https://github.com/pluskid/fitting-random-labels/blob/master/cmd_args.py
# https://github.com/pluskid/fitting-random-labels/blob/master/train.py
class MLP(BaseNNs):
    def __init__(self, dropout_ratio=0.0):
        super(MLP, self).__init__(dropout_ratio)
        self.layer0 = torch.nn.Linear(28*28, 512)
        self.layer1 = torch.nn.Linear(  512, 128)
        self.layer2 = torch.nn.Linear(  128, 512)
        self.layer3 = torch.nn.Linear(  512,  10)

    def forward(self, x):               # := (b, 1,28,28)   # (batch, channel, width, height)
        x = x.flatten(1,-1)             # -> (b, 1*28*28)
        x = torch.relu(self.layer0(x))  # -> (b,200)
        x = torch.relu(self.layer1(x))  # -> (b,200)
        x = self.Dropout(x)
        x = torch.relu(self.layer2(x))  # -> (b,200)
        x = self.Dropout(x)
        return self.layer3(x)           # -> (b,10)
    # restrict parameters
    def clip(self, origin, epsilon):
        c = lambda x,y: torch.max(torch.min(x, y+epsilon), y-epsilon)
        self.layer0.weight.data  = c(self.layer0.weight.data, origin.layer0.weight.data)
        self.layer0.bias.data    = c(self.layer0.bias.data,   origin.layer0.bias.data)
        self.layer1.weight.data  = c(self.layer1.weight.data, origin.layer1.weight.data)
        self.layer1.bias.data    = c(self.layer1.bias.data,   origin.layer1.bias.data)
        self.layer2.weight.data  = c(self.layer2.weight.data, origin.layer2.weight.data)
        self.layer2.bias.data    = c(self.layer2.bias.data,   origin.layer2.bias.data)
        self.layer3.weight.data  = c(self.layer3.weight.data, origin.layer3.weight.data)
        self.layer3.bias.data    = c(self.layer3.bias.data,   origin.layer3.bias.data)

# -------------------------------------------------------------------
