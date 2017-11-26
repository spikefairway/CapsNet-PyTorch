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

from conv_layer import Conv1
from primary_caps import PrimaryCaps
from digit_caps import DigitCaps
from decoder import Decoder


class CapsuleNetwork(nn.Module):
	def __init__(self):
		super(CapsuleNetwork, self).__init__()

		# Build modules for CapsNet.

		## Convolution layer
		self.conv1 = Conv1()

		## PrimaryCaps layer
		self.primary_caps = PrimaryCaps()

		## DigitCaps layer
		self.digit_caps = DigitCaps()

		## Decoder
		self.decoder = Decoder()

	def forward(self, x):
		# x: [bacch_size, 1, 28, 28]

		h = self.conv1(x)
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
		reconstruction_loss = self.reconstruction_loss(images, input, size_average)

		factor = 0.0005

		loss = margin_loss + factor * reconstruction_loss

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
		zero = Variable(torch.zeros(1)).cuda()
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
			L_c = L_c.mean()

		return L_c

	def reconstruction_loss(self, images, input, size_average=True):
		# images: [batch_size, 1, 28, 28]
		# input: [batch_size, 10, 16]

		batch_size = images.size(0)

		# Reconstruct input image.
		reconstructed = self.reconstruct(input)
		# reconstructed: [batch_size, 1, 28, 28]

		# The reconstruction loss is the sum squared difference between the input image and reconstructed image.
		# Multiplied by a small number so it doesn't dominate the margin (class) loss.
		error = (reconstructed - images).view(batch_size, -1)
		error = error**2
		# error: [batch_size, 784=1*28*28]
		error = torch.sum(error, dim=1)
		# error: [batch_size]

		# Average over batch
		if size_average:
			error = error.mean()

		return error

	def reconstruct(self, input, save_path=None):
		# input: [batch_size, 10, 16]

		# Get the lengths of capsule outputs.
		v_mag = torch.sqrt((input**2).sum(dim=2))
		# v_mag: [batch_size, 10]

		# Get index of longest capsule output.
		_, v_max_index = v_mag.max(dim=1)
		v_max_index = v_max_index.data
		# v_max_index: [batch_size]

		# Use just the winning capsule's representation (and zeros for other capsules) to reconstruct input image.
		batch_size = input.size(0)
		all_masked = [None] * batch_size
		for batch_idx in range(batch_size):
			# Get one sample from the batch.
			input_batch = input[batch_idx]
			# input_bacth: [10, 16]

			# Copy only the maximum capsule index from this batch sample.
			# This masks out (leaves as zero) the other capsules in this sample.
			batch_masked = Variable(torch.zeros(input_batch.size())).cuda()
			batch_masked[v_max_index[batch_idx]] = input_batch[v_max_index[batch_idx]]
			# batch_masked: [10, 16]

			all_masked[batch_idx] = batch_masked
		# all_masked: [10, 16] * batch_size

		# Stack masked capsules over the batch dimension.
		masked = torch.stack(all_masked, dim=0)
		# masked: [batch_size, 10, 16]
		masked = masked.view(batch_size, -1)
		# masked: [batch_size, 160]

		# Reconstruct input image.
		reconstructed = self.decoder(masked)
		# reconstructed: [batch_size, 1, 28, 28]

		if save_path is not None:
			output_image = reconstructed.data.cpu()
			# output_image: [batch_size, 1, 28, 28]
			vutils.save_image(output_image, save_path)

		return reconstructed
