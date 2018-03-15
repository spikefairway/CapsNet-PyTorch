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

from .squash import squash


class ConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=9, stride=2):
        super(ConvUnit, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=True
        )

    def forward(self, x):
        # x: [batch_size, in_channels=256, 20, 20]

        h = self.conv(x)
        # h: [batch_size, out_channels=8, 6, 6]

        return h


class PrimaryCaps(nn.Module):
    def __init__(self, in_channels=256, num_capsule_units=32,
                 capsule_size=8, kernel_size=9, stride=2):
        super(PrimaryCaps, self).__init__()

        self.in_channels = in_channels # out_channels of Conv1, a ConvLayer just before PrimaryCaps
        self.capsule_units = num_capsule_units
        self.capsule_size = capsule_size

        # 32 channels of 8D capsules
        self.capsules = ConvUnit(self.in_channels,
                                 self.capsule_units * self.capsule_size,
                                 kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        # x: [batch_size, 256, 20, 20]
        batch_size = x.size(0)
        assert x.size(1) == self.in_channels

        u = self.capsules(x)
        # u: [batch_size, 32 * 8, 6, 6]

        u = u.view(batch_size, capsule_size, -1)
        # u: [batch_size, capsule_size=8, 1152=6*6*32]

        u = u.transpose(1, 2)
        # u: [batch_size, 1152, capsule_size=8]

        u_squashed = squash(u, dim=2)
        # u_squashed: [batch_size, 1152, capsule_size=8]

        return u_squashed
