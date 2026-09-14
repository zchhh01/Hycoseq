import torch

from geometry.geoopt import Lorentz
from geometry.geoopt.manifolds.lorentz import math


class CustomLorentz(Lorentz):
    def __init__(self, k=1.0, learnable=False):
        super(CustomLorentz, self).__init__(k=k, learnable=learnable)

    def sqdist(self, x, y, dim=-1):
        return -2*self.k - 2 * math.inner(x, y, keepdim=False, dim=dim)

    def add_time(self, space):
        time = self.calc_time(space)
        return torch.cat([time, space], dim=-1)

    def calc_time(self, space):
        return torch.sqrt(torch.norm(space, dim=-1, keepdim=True)**2+self.k)

    def centroid(self, x, w=None, eps=1e-8):
        if w is not None:
            avg = w.matmul(x)
        else:
            avg = x.mean(dim=-2)

        denom = (-self.inner(avg, avg, keepdim=True))
        denom = denom.abs().clamp_min(eps).sqrt()

        centroid = torch.sqrt(self.k) * avg / denom

        return centroid

    def switch_man(self, x, manifold_in: Lorentz):
        x = manifold_in.logmap0(x)
        return self.expmap0(x)
    
    def pt_addition(self, x, y):
        z = self.logmap0(y)
        z = self.transp0(x, z)

        return self.expmap(x, z)

    
    
    
    def lorentz_flatten(self, x: torch.Tensor) -> torch.Tensor:







        bs, h, w, c = x.shape

        
        
        space = x.narrow(-1, 1, c - 1).reshape(bs, h * w * (c - 1))

        
        
        time_rescaled = self.calc_time(space)

        return torch.cat([time_rescaled, space], dim=-1)

    def lorentz_reshape_img(self, x: torch.Tensor, img_dim) -> torch.Tensor:
        space = x.narrow(-1, 1, x.shape[-1] - 1)
        space = space.view((-1, img_dim[0], img_dim[1], img_dim[2]-1))
        img = self.add_time(space)

        return img


    
    
    
    def lorentz_relu(self, x: torch.Tensor, add_time: bool=True) -> torch.Tensor:
        return self.lorentz_activation(x, torch.relu, add_time)

    def lorentz_activation(self, x: torch.Tensor, activation, add_time: bool=True) -> torch.Tensor:
        x = activation(x.narrow(-1, 1, x.shape[-1] - 1))
        if add_time:
            x = self.add_time(x)
        return x
    
    def tangent_relu(self, x: torch.Tensor) -> torch.Tensor:
        return self.expmap0(torch.relu(self.logmap0(x)))
