import torch
import torch.nn as nn

__all__ = ['BatchNorm']


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