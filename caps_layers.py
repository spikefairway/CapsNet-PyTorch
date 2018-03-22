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

def squash(x, dim=2):
    v_length_sq = x.pow(2).sum(dim=dim)
    v_length = torch.sqrt(v_length_sq)
    scaling_factor = v_length_sq / (1 + v_length_sq) / v_length

    return x * scaling_factor

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
        self.is_relu = is_relu

        self.capsules = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_capsules * out_capsule_dim,
                kernel_size=kernel_size,
                stride=stride,
                bias=True
        )

    def forward(self, x):
        """
        Revise based on adambielski's implementation.
        ref: https://github.com/adambielski/CapsNet-pytorch/blob/master/net.py
        """
        # x: [batch_size, in_channels=256, 20, 20]
        batch_size = x.size(0)

        out = self.capsules(x)
        if self.is_relu:    # ReLU activation
            out = F.relu(out)
        # out: [batch_size, out_capsules=32 * out_capsule_dim=8 = 256, 6, 6]

        _, C, H, W = out.size()
        out = out.view(batch_size, self.output_caps, self.output_dim, H, W)
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        out = out.view(out.size(0), -1, out.size(4))
        # u: [batch_size, 32 * 6 * 6=1152, 8]

        # Squash vectors
        out = squash(out)

        return out

class DigitCaps(nn.Module):
    def __init__(self, in_capsules, in_capsule_dim, out_capsules, out_capsule_dim,
                 routing_iters=3):
        super(DigitCaps, self).__init__()

        self.routing_iters = routing_iters

        self.in_capsules = in_capsules
        self.in_capsule_dim = in_capsule_dim
        self.out_capsules = out_capsules
        self.out_capsule_dim = out_capsule_dim

        self.W = nn.Parameter(
            torch.Tensor(
                self.in_capsules, 
                self.out_capsules, 
                self.out_capsule_dim, 
                self.in_capsule_dim
            )
        )
        # W: [in_capsules, out_capsules, out_capsule_dim, in_capsule_dim] = [1152, 10, 16, 8]
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_capsules)
        self.W.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: [batch_size, in_capsules=1152, in_capsule_dim=8]
        batch_size = x.size(0)

        x = torch.stack([x] * self.out_capsules, dim=2)
        # x: [batch_size, in_capsules=1152, out_capsules=10, in_capsule_dim=8]

        W = torch.cat([self.W.unsqueeze(0)] * batch_size, dim=0)
        # W: [batch_size, in_capsules=1152, out_capsules=10, out_capsule_dim=16, in_capsule_dim=8]

        # Transform inputs by weight matrix `W`.
        u_hat = torch.matmul(W, x.unsqueeze(4)) # matrix multiplication
        # u_hat: [batch_size, in_capsules=1152, out_capsules=10, out_capsule_dim=16, 1]

        u_hat_detached = u_hat.detach()
        # u_hat_detached: [batch_size, in_capsules=1152, out_capsules=10, out_capsule_dim=16, 1]
        # In forward pass, `u_hat_detached` = `u_hat`, and 
    # in backward, no gradient can flow from `u_hat_detached` back to `u_hat`.

        # Initialize routing logits to zero.
        b_ij = Variable(torch.zeros(batch_size, self.in_capsules, self.out_capsules, 1))
        if torch.cuda.is_available():
            b_ij = b_ij.cuda()
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
