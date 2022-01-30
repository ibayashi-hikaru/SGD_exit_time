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

class quad_func(BaseNNs):
    def __init__(self, num_dim,  k=1.0):
        super(quad_func, self).__init__(k)
        self.k = torch.tensor(k) 
        self.k.requires_grad = False
        self.layer0 = torch.nn.Linear(num_dim, 1, bias=False)

    def init_params(self):
        # Zero init
        with torch.no_grad():
            self.layer0.weight *= 0

    def forward(self, x):
        # Sharpness
        x = torch.pow(self.layer0.weight * torch.sqrt(self.k), 2) 
        x = torch.mean(x)
        return x

class styblinski_tang_func(BaseNNs):
    def __init__(self, num_dim,  k=1.0):
        super(styblinski_tang_func, self).__init__(k)
        self.k = torch.tensor(k) 
        self.k.requires_grad = False
        self.layer0 = torch.nn.Linear(num_dim, 1, bias=False)

    def init_params(self):
        with torch.no_grad():
            self.layer0.weight = torch.nn.Parameter((-2.903534/torch.sqrt(self.k)) * torch.ones((num_dim), device=device))

    def forward(self, x):
        # Sharpness
        term_4 = torch.pow(self.layer0.weight * torch.sqrt(self.k), 4) 
        term_2 = torch.pow(self.layer0.weight * torch.sqrt(self.k), 2) 
        term_1 = self.layer0.weight * torch.sqrt(self.k)
        # There is subtle difference
        x = term_4 - 16 * term_2 + 5 * term_1
        x = torch.mean(x)
        return x

class MLP(BaseNNs):
    def __init__(self, num_dim,  k=1.0):
        super(MLP, self).__init__(k)
        self.k = torch.tensor(k) 
        self.k.requires_grad = False
        self.layer0 = torch.nn.Linear(10, 1000, bias=False)
        self.layer1 = torch.nn.Linear(1000,  2, bias=False)

    def init_params(self):
        self.load_state_dict(self.state_dict(), "./MLP_init_params.pt")
        with torch.no_grad():
            self.layer0.weight /= torch.sqrt(self.k)
            self.layer1.weight /= torch.sqrt(self.k)

    def forward(self, x):
        x = x.flatten(1,-1)
        x = torch.sqrt(self.k) * self.layer0(x)
        x = torch.relu(x) 
        x = torch.sqrt(self.k) * self.layer1(x) 
        # Emulating quadratic loss as if every label is 0
        x = x.norm(dim=1, p=2)
        x = torch.mean(x)
        return x
