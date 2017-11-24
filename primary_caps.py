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

from utils import squash


class ConvUnit(nn.Module):
	def __init__(self):
		super(ConvUnit, self).__init__()

		self.conv0 = nn.Conv2d(
			in_channels=256,
			out_channels=8,
			kernel_size=9,
			stride=2,
			bias=True
		)

	def forward(self, x):
		return self.conv0(x)


class PrimaryCaps(nn.Module):
	def __init__(self):
		super(PrimaryCaps, self).__init__()

		

		def create_conv_unit(unit_idx):
				unit = ConvUnit()
				self.add_module("unit_" + str(unit_idx), unit)
				return unit

		self.units = [create_conv_unit(i) for i in range(32)]

	def forward(self, x):
		batch_size = x.size(0)

		u = []
		for i in range(32):
			u_i = self.units[i](x)

			# (batch_size, 8, 36, 1)
			u_i = u_i.view(batch_size, 8, -1, 1)

			u.append(u_i)
		# u: (batch_size, 8, 36, 1) x 32

		# (batch_size, 8, 36, 32)
		u = torch.cat(u, dim=3)

		# Flatten to (batch_size, 8, 1152=36*32).
		u = u.view(batch_size, 8, -1)

		# Transpose to (batch_size, 1152, 8)
		u = u.transpose(1, 2)

		# Return squashed outputs.
		return squash(u, dim=2)
