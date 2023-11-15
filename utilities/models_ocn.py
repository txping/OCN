import torch
import torch.nn as nn

from .utils import *


class OCN(nn.Module):

    def __init__(self, args):
        super(OCN, self).__init__()

        d = len(args.x0)

        self.net = nn.ModuleList()
        self.net.append(nn.Linear(d, args.hidden_neurons))
        self.net.append(nn.Tanh())
        for i in range(1, args.hidden_layers):
            self.net.append(nn.Linear(args.hidden_neurons, args.hidden_neurons))
            self.net.append(nn.Tanh())
        self.net.append(nn.Linear(args.hidden_neurons, d))

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def F(self, x):
        for f in self.net:
            x = f(x)
        return x
    
    def forward(self, t, x):
        for f in self.net:
            x = f(x)
        return x



class OCN_GF(nn.Module):

    def __init__(self, args):
        super(OCN_GF, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, args.hidden_neurons),
            nn.Tanh(),
            nn.Linear(args.hidden_neurons, args.hidden_neurons),
            nn.Tanh(),
            nn.Linear(args.hidden_neurons, 1),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def F(self, x):
        return torch.squeeze(self.net(x))
        # the output dimension is 1

    def forward(self, t, x):
        F = self.net(x)
        dFdx = torch.autograd.grad(outputs=F, inputs=x, grad_outputs=torch.ones_like(F), create_graph=True)[0]
        return dFdx




