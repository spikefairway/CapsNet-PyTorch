#
# Dynamic Routing Between Capsules
# https://arxiv.org/pdf/1710.09829.pdf
#

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F

from squash import squash

class PrimaryCaps(nn.Module):
    """
    PrimaryCaps layers.
    """
    def __init__(self, in_channels, out_capsules, out_capsule_dim,
                 kernel_size=9, stride=2, is_relu=False):
        super(PrimaryCaps, self).__init__()

        self.in_channels = in_channels
        self.out_capsules = out_capsules
        self.out_capsule_dim = out_capsule_dim

        self.capsules = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_capsules * out_capsule_dim,
                kernel_size=kernel_size,
                stride=stride,
                bias=True
        )

    def forward(self, x):
        # x: [batch_size, 256, 20, 20]
        batch_size = x.size(0)

        u = self.capsules(x)
        # u: [batch_size, 8*32, 6, 6]

        u = u.view(batch_size, self.out_capsule_dim, -1)
        # u: [batch_size, out_capsule_dim=8, 1152=6*6*32]

        u = u.transpose(1, 2)
        # u: [batch_size, 1152, out_capsule_dim=8]

        u_squashed = squash(u, dim=2)
        # u_squashed: [batch_size, 1152, out_capsule_dim=8]

        return u_squashed

class DigitCaps(nn.Module):
    def __init__(self, routing_iters, gpu):
        super(DigitCaps, self).__init__()

        self.routing_iters = routing_iters
        self.gpu = gpu

        self.in_capsules = 1152
        self.in_out_capsule_dim = 8
        self.out_capsules = 10
        self.out_capsule_dim = 16

        self.W = nn.Parameter(
            torch.Tensor(
                self.in_capsules, 
                self.out_capsules, 
                self.out_capsule_dim, 
                self.in_out_capsule_dim
            )
        )
        # W: [in_capsules, out_capsules, out_capsule_dim, in_out_capsule_dim] = [1152, 10, 16, 8]
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_capsules)
        self.W.data.uniform_(-stdv, stdv)

    # FIXME, write in an easier way to understand, some tensors have some redundant dimensions.
    def forward(self, x):
        # x: [batch_size, in_capsules=1152, in_out_capsule_dim=8]
        batch_size = x.size(0)

        x = torch.stack([x] * self.out_capsules, dim=2)
        # x: [batch_size, in_capsules=1152, out_capsules=10, in_out_capsule_dim=8]

        W = torch.cat([self.W.unsqueeze(0)] * batch_size, dim=0)
        # W: [batch_size, in_capsules=1152, out_capsules=10, out_capsule_dim=16, in_out_capsule_dim=8]

        # Transform inputs by weight matrix `W`.
        u_hat = torch.matmul(W, x.unsqueeze(4)) # matrix multiplication
        # u_hat: [batch_size, in_capsules=1152, out_capsules=10, out_capsule_dim=16, 1]

        u_hat_detached = u_hat.detach()
        # u_hat_detached: [batch_size, in_capsules=1152, out_capsules=10, out_capsule_dim=16, 1]
        # In forward pass, `u_hat_detached` = `u_hat`, and 
    # in backward, no gradient can flow from `u_hat_detached` back to `u_hat`.

        # Initialize routing logits to zero.
        b_ij = Variable(torch.zeros(batch_size, self.in_capsules, self.out_capsules, 1))
        if self.gpu >= 0:
            b_ij = b_ij.cuda(self.gpu)
        # b_ij: [batch_size, in_capsules=1152, out_capsules=10, 1]

        # Iterative routing.
        for iteration in range(self.routing_iters):
            # Convert routing logits to softmax.
            c_ij = F.softmax(b_ij, dim=2).unsqueeze(4)
            # c_ij: [batch_size, in_capsules=1152, out_capsules=10, 1, 1]

            if iteration == self.routing_iters - 1:
                # Apply routing `c_ij` to weighted inputs `u_hat`.
                s_j = (c_ij * u_hat).sum(dim=1, keepdim=True) # element-wise product
                # s_j: [batch_size, 1, out_capsules=10, out_capsule_dim=16, 1]
    
                v_j = squash(s_j, dim=3)
                # v_j: [batch_size, 1, out_capsules=10, out_capsule_dim=16, 1]

            else:
                # Apply routing `c_ij` to weighted inputs `u_hat`.
                s_j = (c_ij * u_hat_detached).sum(dim=1, keepdim=True) # element-wise product
                # s_j: [batch_size, 1, out_capsules=10, out_capsule_dim=16, 1]
    
                v_j = squash(s_j, dim=3)
                # v_j: [batch_size, 1, out_capsules=10, out_capsule_dim=16, 1]
    
                # Compute inner products of 2 16D-vectors, `u_hat` and `v_j`.
                u_vj1 = torch.matmul(u_hat_detached.transpose(3, 4), v_j).squeeze(4)
                # u_vj1: [batch_size, in_capsules=1152, out_capsules=10, 1]
                # Not calculate batch mean.
    
                # Update b_ij (routing).
                b_ij = b_ij + u_vj1

        return v_j.squeeze(4).squeeze(1) # [batch_size, out_capsules=10, out_capsule_dim=16]