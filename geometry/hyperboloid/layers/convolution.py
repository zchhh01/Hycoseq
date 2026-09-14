import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from geometry.hyperboloid.geometry import CustomLorentz
from geometry.hyperboloid.layers.projection import LorentzFullyConnected

class LorentzConv1d(nn.Module):






    def __init__(
            self,
            manifold: CustomLorentz,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            bias=True,
            LFC_normalize=False
    ):
        super(LorentzConv1d, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        lin_features = (self.in_channels - 1) * self.kernel_size + 1

        self.linearized_kernel = LorentzFullyConnected(
            manifold,
            lin_features, 
            self.out_channels, 
            bias=bias,
            normalize=LFC_normalize
        )

    def forward(self, x):
        bsz = x.shape[0]

        
        x = F.pad(x, (0, 0, self.padding, self.padding))
        x[..., 0].clamp_(min=self.manifold.k.sqrt()) 

        patches = x.unfold(1, self.kernel_size, self.stride)
        
        patches_time = patches.narrow(2, 0, 1)
        patches_time_rescaled = torch.sqrt(torch.sum(patches_time ** 2, dim=(-2,-1), keepdim=True) - ((self.kernel_size - 1) * self.manifold.k))
        patches_time_rescaled = patches_time_rescaled.view(bsz, patches.shape[1], -1)

        patches_space = patches.narrow(2, 1, patches.shape[2]-1).reshape(bsz, patches.shape[1], -1)
        patches_pre_kernel = torch.concat((patches_time_rescaled, patches_space), dim=-1)

        out = self.linearized_kernel(patches_pre_kernel)

        return out


class LorentzConv2d(nn.Module):






    def __init__(
            self,
            manifold: CustomLorentz,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            LFC_normalize=False
    ):
        super(LorentzConv2d, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation

        self.kernel_len = self.kernel_size[0] * self.kernel_size[1]

        lin_features = ((self.in_channels - 1) * self.kernel_size[0] * self.kernel_size[1]) + 1

        self.linearized_kernel = LorentzFullyConnected(
            manifold,
            lin_features, 
            self.out_channels, 
            bias=bias,
            normalize=LFC_normalize
        )
        self.unfold = torch.nn.Unfold(kernel_size=(self.kernel_size[0], self.kernel_size[1]), dilation=dilation, padding=padding, stride=stride)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = math.sqrt(2.0 / ((self.in_channels-1) * self.kernel_size[0] * self.kernel_size[1]))
        self.linearized_kernel.weight.weight.data.uniform_(-stdv, stdv)
        if self.bias:
            self.linearized_kernel.weight.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        bsz = x.shape[0]
        h, w = x.shape[1:3]

        h_out = math.floor(
            (h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = math.floor(
            (w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        x = x.permute(0, 3, 1, 2)

        patches = self.unfold(x)  
        patches = patches.permute(0, 2, 1)

        
        patches_time = torch.clamp(patches.narrow(-1, 0, self.kernel_len), min=self.manifold.k.sqrt())  
        patches_time_rescaled = torch.sqrt(torch.sum(patches_time ** 2, dim=-1, keepdim=True) - ((self.kernel_len - 1) * self.manifold.k))

        patches_space = patches.narrow(-1, self.kernel_len, patches.shape[-1] - self.kernel_len)
        patches_space = patches_space.reshape(patches_space.shape[0], patches_space.shape[1], self.in_channels - 1, -1).transpose(-1, -2).reshape(patches_space.shape) 

        patches_pre_kernel = torch.concat((patches_time_rescaled, patches_space), dim=-1)

        out = self.linearized_kernel(patches_pre_kernel)
        out = out.view(bsz, h_out, w_out, self.out_channels)

        return out

class LorentzConvTranspose2d(nn.Module):






    def __init__(
            self, 
            manifold: CustomLorentz, 
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=1, 
            padding=0, 
            output_padding=0, 
            bias=True,
            LFC_normalize=False
        ):
        super(LorentzConvTranspose2d, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        if isinstance(output_padding, int):
            self.output_padding = (output_padding, output_padding)
        else:
            self.output_padding = output_padding

        padding_implicit = [0,0]
        padding_implicit[0] = kernel_size - self.padding[0] - 1 
        padding_implicit[1] = kernel_size - self.padding[1] - 1 

        self.pad_weight = nn.Parameter(F.pad(torch.ones((self.in_channels,1,1,1)),(1,1,1,1)), requires_grad=False)

        self.conv = LorentzConv2d(
            manifold=manifold, 
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=padding_implicit, 
            bias=bias, 
            LFC_normalize=LFC_normalize
        )

    def forward(self, x):
        if self.stride[0] > 1 or self.stride[1] > 1:
            
            x = x.permute(0,3,1,2)
            
            x = F.conv_transpose2d(x, self.pad_weight,stride=self.stride,padding=1, groups=self.in_channels)
            x = x.permute(0,2,3,1)
            x[..., 0].clamp_(min=self.manifold.k.sqrt())

        x = self.conv(x)

        if self.output_padding[0] > 0 or self.output_padding[1] > 0:
            x = F.pad(x, pad=(0, self.output_padding[1], 0, self.output_padding[0])) 
            x[..., 0].clamp_(min=self.manifold.k.sqrt()) 

        return x
