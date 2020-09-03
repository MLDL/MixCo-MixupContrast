# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from builders.utils import * 

__all__ = ['MixCo']


class MixCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.2, mix_T=0.2, mix_param=0.1, mlp=False, single_gpu=False, small_input=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MixCo, self).__init__()
        
        self.single_gpu = single_gpu

        self.K = K
        self.m = m
        self.T = T
        self.mix_T = mix_T
        self.mix_param = mix_param
        
        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim, small_input=small_input)
        self.encoder_k = base_encoder(num_classes=dim, small_input=small_input)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.soft_loss = SoftCrossEntropy()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        if not self.single_gpu:
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        imgs_mix, lbls_mix = self.img_mixer(im_q)

        # compute query features
        q = self.encoder_q(torch.cat((im_q, imgs_mix)))  # queries: (N+N/2)xC
        q = nn.functional.normalize(q, dim=1)

        q_mix = q[im_q.size(0):]
        q = q[:im_q.size(0)]

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            if not self.single_gpu:
                im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            if not self.single_gpu:
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        # mixed logits: N/2 x N
        logits_mix_pos = torch.mm(q_mix, k.transpose(0, 1)) 
        # mixed negative logits: N/2 x K
        logits_mix_neg = torch.mm(q_mix, self.queue.clone().detach())
        logits_mix = torch.cat([logits_mix_pos, logits_mix_neg], dim=1) # N/2 x (N+K)

        lbls_mix = torch.cat([lbls_mix, torch.zeros_like(logits_mix_neg)], dim=1)

        # apply temperature
        logits /= self.T
        logits_mix /= self.mix_T

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels, logits_mix, lbls_mix
   
    def rep_mixer(self, rep_q, rep_k, neg_queue, mix_size=10):
        with torch.no_grad():
            device = rep_k.device
            b, dim = rep_k.size(0), rep_k.size(1)

            lam = np.random.uniform(0, 1, size=(b, mix_size, 1))
            lam = torch.from_numpy(lam).float().to(device)
            idx = np.random.randint(0, neg_queue.size(0), size=(b, mix_size))
            idx = torch.from_numpy(idx).long().to(device)

            rep_elem = rep_k.unsqueeze(1).expand(b, mix_size, dim)
            rep_mix = lam * rep_elem + (1-lam) * neg_queue[idx]

        logits = torch.einsum('nc,nkc->nk', [rep_q, rep_mix])
        labels = lam.squeeze()

        return logits, labels
    
    @torch.no_grad()
    def img_mixer(self, im_k):
        B = im_k.size(0)
        assert B % 2 == 0
        sid = int(B/2)
        im_k1, im_k2 = im_k[:sid], im_k[sid:]
        lam = torch.from_numpy(np.random.uniform(0, 1, size=(sid,1,1,1))).float().to(im_k.device)
        imgs_mix = lam * im_k1 + (1-lam) * im_k2
        lbls_mix = torch.cat((torch.diag(lam.squeeze()), torch.diag((1-lam).squeeze())), dim=1)
        
        return imgs_mix, lbls_mix
