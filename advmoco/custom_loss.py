import torch
import torch.nn as nn


class NCEwithPerturbLoss(nn.Module):
    def __init__(self, T=0.2):
        super(NCEwithPerturbLoss, self).__init__()
        self.T = T
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, q, k, neg, epsilon, labels, neg_length=None):
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        # negative logits: NxK
        assert q.size() == epsilon.size()
        if neg_length is not None:
            perm = torch.randperm(neg.size(1))[:neg_length]
            neg = neg[:,perm]
        l_neg = torch.einsum('nc,ck->nk', [q + epsilon, neg])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T
        
        return self.criterion(logits, labels)
