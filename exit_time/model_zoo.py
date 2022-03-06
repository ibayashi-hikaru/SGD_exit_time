import copy
import torch
class Base(torch.nn.Module):
    def __init__(self):
        super(Base, self).__init__()

    def get_param(self):
        state = copy.deepcopy(self.state_dict())
        for k in state:
            state[k] = state[k].cpu().detach().numpy()
        return state

    def set_param(self, param):
        for k in param:
            param[k] = torch.tensor(param[k],dtype=torch.float)
        self.load_state_dict(param) 

class MLP(BaseNNs):
    def __init__(self, dropout_ratio=0.0):
        super(MLP, self).__init__(dropout_ratio)
        self.layer0 = torch.nn.Linear(28*28, 512)
        self.layer1 = torch.nn.Linear(  512, 1)

    def forward(self, x):               # := (b, 1,28,28)   # (batch, channel, width, height)
        x = x.flatten(1,-1)             # -> (b, 1*28*28)
        x = torch.relu(self.layer0(x))  # -> (b, 512)
        return self.layer1(x)           # -> (b,1)
