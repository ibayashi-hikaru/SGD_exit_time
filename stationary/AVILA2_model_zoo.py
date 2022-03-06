import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float
# -------------------------------------------------------------------
import copy
class BaseNNs(torch.nn.Module):
    def __init__(self, k):
        super(BaseNNs, self).__init__()

    def get_param(self):
        state = copy.deepcopy(self.state_dict())
        for k in state:
            state[k] = state[k].cpu().detach().numpy()
        return state

    def update(self, lr):
        with torch.no_grad():
            for param in self.parameters():
                param -= lr*param.grad

    def perturb(self, lr):
        with torch.no_grad():
            for param in self.parameters():
                param -= lr*torch.randn(param.size(), device=device)

    def set_param(self, param):
        for k in param:
            param[k] = torch.tensor(param[k],dtype=torch.float)
        self.load_state_dict(param) 

# -------------------------------------------------------------------

class styblinski_tang_func(BaseNNs):
    def __init__(self, num_dim,  k=1.0):
        super(styblinski_tang_func, self).__init__(k)
        self.layer0 = torch.nn.Linear(num_dim, 1, bias=False)

    def init_params(self):
        # Zero init
        with torch.no_grad():
            self.layer0.weight *= 0

    def forward(self):
        term_4 = torch.pow(self.layer0.weight, 4) 
        term_2 = torch.pow(self.layer0.weight, 2) 
        term_1 = self.layer0.weight
        # There is subtle difference
        x = term_4 - 16 * term_2 + 5 * term_1
        x = torch.mean(x)
        return x

class quad_func(BaseNNs):
    def __init__(self, num_dim,  k=1.0):
        super(quad_func, self).__init__(k)
        self.layer0 = torch.nn.Linear(num_dim, 1, bias=False)

    def init_params(self):
        # Zero init
        with torch.no_grad():
            self.layer0.weight *= 0

    def forward(self):
        x = torch.pow(self.layer0.weight, 2) 
        x = torch.mean(x)
        return x
