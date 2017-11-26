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


class ConvUnit(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(ConvUnit, self).__init__()

		self.conv = nn.Conv2d(
			in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=9,
			stride=2,
			bias=True
		)

	def forward(self, x):
		# x: [batch_size, in_channels=256, 20, 20]

		h = self.conv(x)
		# h: [batch_size, out_channels=8, 6, 6]

		return h


class PrimaryCaps(nn.Module):
	def __init__(self):
		super(PrimaryCaps, self).__init__()

		self.conv1_out = 256 # out_channels of Conv1, a ConvLayer just before PrimaryCaps
		self.capsule_units = 32
		self.capsule_size = 8

		def create_conv_unit(unit_idx):
				unit = ConvUnit(
					in_channels=self.conv1_out, 
					out_channels=self.capsule_size
				)
				self.add_module("unit_" + str(unit_idx), unit)
				return unit

		self.conv_units = [create_conv_unit(i) for i in range(self.capsule_units)]

	def forward(self, x):
		# x: [batch_size, 256, 20, 20]
		batch_size = x.size(0)

		u = []
		for i in range(self.capsule_units):
			u_i = self.conv_units[i](x)
			# u_i: [batch_size, capsule_size=8, 6, 6]

			u_i = u_i.view(batch_size, self.capsule_size, -1, 1)
			# u_i: [batch_size, capsule_size=8, 36, 1]

			u.append(u_i)
		# u: [batch_size, capsule_size=8, 36, 1] x capsule_units=32

		u = torch.cat(u, dim=3)
		# u: [batch_size, capsule_size=8, 36, capsule_units=32]

		u = u.view(batch_size, self.capsule_size, -1)
		# u: [batch_size, capsule_size=8, 1152=36*32]

		u = u.transpose(1, 2)
		# u: [batch_size, 1152, capsule_size=8]

		u_squashed = squash(u, dim=2)
		# u_squashed: [batch_size, 1152, capsule_size=8]

		return u_squashed
