import torch
import torch.nn as nn

from geometry.hyperboloid.geometry import CustomLorentz

class LorentzAct(nn.Module):

    def __init__(self, activation, manifold: CustomLorentz):
        super(LorentzAct, self).__init__()
        self.manifold = manifold
        self.activation = activation 

    def forward(self, x):
        return self.manifold.lorentz_activation(x, self.activation)
    

class LorentzReLU(nn.Module):

    def __init__(self, manifold: CustomLorentz):
        super(LorentzReLU, self).__init__()
        self.manifold = manifold

    def forward(self, x):
        return self.manifold.lorentz_relu(x)


class LorentzGlobalAvgPool2d(torch.nn.Module):

    def __init__(self, manifold: CustomLorentz, keep_dim=False):
        super(LorentzGlobalAvgPool2d, self).__init__()

        self.manifold = manifold
        self.keep_dim = keep_dim

    def forward(self, x):
        bs, h, w, c = x.shape
        x = x.view(bs, -1, c)
        x = self.manifold.centroid(x)
        if self.keep_dim:
            x = x.view(bs, 1, 1, c)

        return x
