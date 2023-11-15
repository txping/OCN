import numpy as np
import torch
import torch.nn as nn


class LGF(nn.Module):
    def __init__(self):
        super(LGF, self).__init__()
        self.A = torch.tensor([[-2., -1.], [-1., -2.]])

    def F(self, y):
        y1, y2 = y[..., 0], y[..., 1]
        return (self.A[0][0]/2) * y1**2 + (self.A[1][1]/2) * y2**2 + self.A[0][1] * y1*y2

    def dFdy(self, y):
        return torch.mm(y, self.A)

    def forward(self, t, y):
        return self.dFdy(y)



class NGF(nn.Module):
    def __init__(self):
        super(NGF, self).__init__()

    def F(self, y):
        y1, y2 = y[..., 0], y[..., 1]
        return np.sin(y1) * np.cos(y2)

    def dFdy(self, y):
        dF = torch.zeros_like(y)
        y1, y2 = y[..., 0], y[..., 1]
        dF[..., 0] = - np.cos(y1) * np.cos(y2)
        dF[..., 1] = np.sin(y1) * np.sin(y2)
        return dF

    def forward(self, t, y):
        return self.dFdy(y)


class Pendulum(nn.Module):
    def __init__(self):
        super(Pendulum, self).__init__()

    def F(self, y):
        # y.shape = number of y0 x dimension of y0
        f = torch.zeros_like(y)
        y1, y2 = y[...,0], y[...,1]
    
        f[...,0] = y2
        f[...,1] = - 0.2*y2 - 8.91*np.sin(y1)
        return f

    def forward(self, t, y):
        return self.F(y)


class Lorenz(nn.Module):
    def __init__(self):
        super(Lorenz, self).__init__()

    def F(self, y):
        # y.shape = number of y0 x dimension of y0
        f = torch.zeros_like(y)
        y1, y2, y3 = y[..., 0], y[..., 1], y[..., 2]
    
        f[...,0] = 10 * (y2 - y1)
        f[...,1] = y1 * (28 - y3) - y2
        f[...,2] = y1 * y2 - (8/3) * y3
        return f

    def forward(self, t, y):
        return self.F(y)
        


   