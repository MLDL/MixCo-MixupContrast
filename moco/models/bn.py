import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ['BatchNorm', 'LayerNorm']


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.9):
        super().__init__(num_features, eps, momentum)
        self.mean_array = []
        self.var_array = []
        
        
    def forward(self, x):
        if self.training is True:
            mini_batch_mean = torch.mean(x, dim=(0, 2, 3)).detach()
            mini_batch_var = torch.var(x, dim=(0, 2, 3)).detach()
            self.mean_array.append(round(torch.norm(mini_batch_mean).data.cpu().item(), 4))
            self.var_array.append(round(torch.norm(mini_batch_var).data.cpu().item(), 4))
        
        return super().forward(x)

class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
    
    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input=Tensor):
        self.normalized_shape = tuple(input.size()[1:])
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*self.normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*self.normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
