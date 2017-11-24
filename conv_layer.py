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


class Conv1(nn.Module):
	def __init__(self):
		'''
		`Conv1` is a ordinary 2D convolutional layer with 9x9 kernels, 
		stride 2, 256 output channels, and ReLU activations.
		'''
		super(Conv1, self).__init__()

		self.conv0 = nn.Conv2d(
			in_channels=1,
			out_channels=256,
			kernel_size=9,
			stride=1,
			bias=True
		)

		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		return self.relu(self.conv0(x))
