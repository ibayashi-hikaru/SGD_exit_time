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

    # restrict parameters
    def clip(self, origin, epsilon):
        c = lambda x,y: torch.max(torch.min(x, y+epsilon), y-epsilon)
        self.layer0.weight.data  = c(self.layer0.weight.data, origin.layer0.weight.data)
        self.layer1.weight.data  = c(self.layer1.weight.data, origin.layer1.weight.data)
        self.layer2.weight.data  = c(self.layer2.weight.data, origin.layer2.weight.data)
        self.layer3.weight.data  = c(self.layer3.weight.data, origin.layer3.weight.data)
        self.layer4.weight.data  = c(self.layer4.weight.data, origin.layer4.weight.data)

# -------------------------------------------------------------------
class VGG(BaseNNs):
    def __init__(self,dropout_ratio=0.0):
        super(VGG, self).__init__(dropout_ratio)
        self.layer0 = torch.nn.Conv2d(  3, 16, kernel_size=3)
        self.layer1 = torch.nn.Conv2d( 16, 16, kernel_size=3)
        self.layer2 = torch.nn.Conv2d( 16, 32, kernel_size=3)
        self.layer3 = torch.nn.Conv2d( 32, 64, kernel_size=3, padding=1)
        self.layer4 = torch.nn.Conv2d( 64, 64, kernel_size=3, padding=2)
        self.layer5 = torch.nn.Linear( 64, 128)
        self.layer6 = torch.nn.Linear(128, 2)

    def forward(self, x):                               # := (b,  3,32,32)  # (batch, channel, width, height)
        x = torch.relu(self.layer0(x))                  # -> (b, 16,32,32)
        x = torch.nn.functional.max_pool2d(x, 2, 2)     # -> (b, 16,16,16) 
        x = torch.relu(self.layer1(x))                  # -> (b, 16,16,16)
        x = torch.nn.functional.max_pool2d(x, 2, 2)     # -> (b,  8, 8,16)
        x = torch.relu(self.layer2(x))                  # -> (b,  8, 8,32)
        x = torch.nn.functional.max_pool2d(x, 2, 2)     # -> (b,  4, 4,32)
        x = torch.relu(self.layer3(x))                  # -> (b,  4, 4,64)
        x = torch.nn.functional.max_pool2d(x, 2, 2)     # -> (b,  2, 2,64)
        x = torch.relu(self.layer4(x))                  # -> (b,  2, 2,64)
        x = torch.nn.functional.max_pool2d(x, 2, 2)     # -> (b,  1, 1,64)
        x = x.flatten(1,-1)                             # -> (b, 64)
        x = torch.relu(self.layer5(x))                  # -> (b,128)
        return self.layer6(x)                           # -> (b,10)

    # restrict parameters
    def clip(self, origin, epsilon):
        c = lambda x,y: torch.max(torch.min(x, y+epsilon), y-epsilon)
        self.layer0.weight.data  = c(self.layer0.weight.data, origin.layer0.weight.data)
        self.layer1.weight.data  = c(self.layer1.weight.data, origin.layer1.weight.data)
        self.layer2.weight.data  = c(self.layer2.weight.data, origin.layer2.weight.data)
        self.layer3.weight.data  = c(self.layer3.weight.data, origin.layer3.weight.data)
        self.layer4.weight.data  = c(self.layer4.weight.data, origin.layer4.weight.data)
        self.layer5.weight.data  = c(self.layer5.weight.data, origin.layer5.weight.data)
        self.layer6.weight.data  = c(self.layer6.weight.data, origin.layer6.weight.data)

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
