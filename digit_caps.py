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


class DigitCaps(nn.Module):
	def __init__(self):
		super(DigitCaps, self).__init__()

		# (in_capsules, out_capsules, out_capsule_size, in_capsule_size) = (1152, 10, 16, 8)
		self.W = nn.Parameter(torch.randn(1152, 10, 16, 8))

	# FIXME, write in a easier way to understand, some tensors have some extra dimensions.
	def forward(self, x):
		batch_size = x.size(0)

		# (batch_size, 1152, 8) -> (batch_size, 1152, 10, 8)
		x = torch.stack([x] * 10, dim=2)

		# (batch_size, 1152, 10, 16, 8)
		W = torch.cat([self.W.unsqueeze(0)] * batch_size, dim=0)

		# Transform inputs by weight matrix.
		# (batch_size, 1152, 10, (16, 1))
		u_hat = torch.matmul(W, x.unsqueeze(4)) # matrix multiplication

		# Initialize routing logits to zero.
		# (1152, 10, 1)
		b_ij = Variable(torch.zeros(1152, 10, 1)).cuda()

		# Iterative routing.
		num_iterations = 3
		for iteration in range(num_iterations):
			# Convert routing logits to softmax.
			# (batch_size, 1152, 10, 1, 1)
			c_ij = F.softmax(b_ij.unsqueeze(0))
			c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

			# Apply routing (c_ij) to weighted inputs (u_hat).
			# (batch_size, 1, 10, (16, 1))
			s_j = (c_ij * u_hat).sum(dim=1, keepdim=True) # element-wise product

			# (batch_size, 1, 10, (16, 1))
			v_j = squash(s_j, dim=3)

			# u_vj1 = (1152, 10, 1), computed by inner products of 16D vectors
			u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j).squeeze(4).mean(dim=0, keepdim=False)

			# Update b_ij (routing)
			b_ij = b_ij + u_vj1

		return v_j.squeeze(1) # (batch_size, 10, 16, 1)
