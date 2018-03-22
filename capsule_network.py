#
# Dynamic Routing Between Capsules
# https://arxiv.org/pdf/1710.09829.pdf
#

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.utils as vutils
import torch.nn.functional as F

from caps_layers import PrimaryCaps, DigitCaps
from decoder import Decoder


class CapsuleNetwork(nn.Module):
    def __init__(self, routing_iters, is_relu=False):
        super(CapsuleNetwork, self).__init__()

        # Build modules for CapsNet.

        ## Convolution layer
        self.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=256,
                kernel_size=9,
                stride=1,
                bias=True
        )

        ## PrimaryCaps layer
        self.primary_caps = PrimaryCaps(256, 32, 8, is_relu=is_relu)

        ## DigitCaps layer
        self.digit_caps = DigitCaps(32, 8, 10, 16, routing_iters=routing_iters)

        ## Decoder
        self.decoder = Decoder()

    def forward(self, x):
        # x: [batch_size, 1, 28, 28]

        h = F.relu(self.conv1(x))
        # h: [batch_size, 256, 20, 20]

        h = self.primary_caps(h)
        # h: [batch_size, 1152=primary_capsules, 8=primary_capsule_size]

        h = self.digit_caps(h)
        # h: [batch_size, 10=digit_capsule, 16=digit_capsule_size]

        return h

    def loss(self, images, input, target, size_average=True):
        # images: [batch_size, 1, 28, 28]
        # input: [batch_size, 10, 16, 1]
        # target: [batch_size, 10]

        margin_loss = self.margin_loss(input, target, size_average)
        reconstruction_loss = self.reconstruction_loss(images, input, target, size_average)

        loss = margin_loss + reconstruction_loss

        return loss, margin_loss, reconstruction_loss

    def margin_loss(self, input, target, size_average=True):
        # images: [batch_size, 1, 28, 28]
        # input: [batch_size, 10, 16]
        # target: [batch_size, 10]

        batch_size = input.size(0)

        # ||vc|| from the paper.
        v_mag = torch.sqrt((input**2).sum(dim=2, keepdim=True))
        # v_mag: [batch_size, 10, 1]

        # Calculate left and right max() terms from Eq.4 in the paper.
        zero = Variable(torch.zeros(1))
        if torch.cuda.is_available():
            zero = zero.cuda()
        m_plus = 0.9
        m_minus = 0.1
        max_l = torch.max(m_plus - v_mag, zero).view(batch_size, -1)**2
        max_r = torch.max(v_mag - m_minus, zero).view(batch_size, -1)**2
        # max_l, max_r: [batch_size, 10]

        # This is Eq.4 from the paper.
        loss_lambda = 0.5
        T_c = target
        # T_c: [batch_size, 10]
        L_c = T_c * max_l + loss_lambda * (1.0 - T_c) * max_r
        # L_c: [batch_size, 10]
        L_c = L_c.sum(dim=1)
        # L_c: [batch_size]

        if size_average:
            L_c = L_c.mean() # average over batch.
        else:
            L_c = L_c.sum() # sum over batch.

        return L_c

    def reconstruction_loss(self, images, input, target, size_average=True):
        # images: [batch_size, 1, 28, 28]
        # input: [batch_size, 10, 16]
        # target: [batch_size, 10]

        batch_size = images.size(0)

        # Reconstruct input image.
        reconstructed = self.reconstruct(input, target)
        # reconstructed: [batch_size, 1, 28, 28]

        # The reconstruction loss is the sum squared difference between the input image and reconstructed image.
        # Multiplied by a small number so it doesn't dominate the margin (class) loss.
        error = (reconstructed - images).view(batch_size, -1)
        error = error**2
        # error: [batch_size, 784=1*28*28]
        error = torch.sum(error, dim=1)
        # error: [batch_size]

        if size_average:
            error = error.mean() # average over batch.
        else:
            error = error.sum() # sum over batch.

        rec_loss_weight = 0.0005
        error *= rec_loss_weight

        return error

    def reconstruct(self, input, target):
        # input: [batch_size, 10, 16]
        # target: [batch_size, 10]
        batch_size = input.size(0)

        # Mask with true label
        mask0 = target.unsqueeze(2)
        mask = torch.stack([mask0] * input.size(2), dim=2)
                # mask: [batch_size, 10, 16]

        # Stack masked capsules over the batch dimension.
        masked = input * mask.squeeze(3)
        # masked: [batch_size, 10, 16]
        masked = masked.view(batch_size, -1)
        # masked: [batch_size, 160]

        # Reconstruct input image.
        reconstructed = self.decoder(masked)
        # reconstructed: [batch_size, 1, 28, 28]

        return reconstructed
