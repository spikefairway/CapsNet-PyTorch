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


class DigitCaps(nn.Module):
	def __init__(self):
		super(DigitCaps, self).__init__()

		self.in_capsules = 1152
		self.in_capsule_size = 8
		self.out_capsules = 10
		self.out_capsule_size = 16

		self.W = nn.Parameter(
			torch.randn(
				self.in_capsules, 
				self.out_capsules, 
				self.out_capsule_size, 
				self.in_capsule_size
			)
		)
		# W: [in_capsules, out_capsules, out_capsule_size, in_capsule_size] = [1152, 10, 16, 8]

	# FIXME, write in an easier way to understand, some tensors have some extra dimensions.
	def forward(self, x):
		# x: [batch_size, in_capsules=1152, in_capsule_size=8]
		batch_size = x.size(0)

		x = torch.stack([x] * self.out_capsules, dim=2)
		# x: [batch_size, in_capsules=1152, out_capsules=10, in_capsule_size=8]

		W = torch.cat([self.W.unsqueeze(0)] * batch_size, dim=0)
		# W: [batch_size, in_capsules=1152, out_capsules=10, out_capsule_size=16, in_capsule_size=8]

		# Transform inputs by weight matrix `W`.
		u_hat = torch.matmul(W, x.unsqueeze(4)) # matrix multiplication
		# u_hat: [batch_size, in_capsules=1152, out_capsules=10, out_capsule_size=16, 1]

		# Initialize routing logits to zero.
		b_ij = Variable(torch.zeros(self.in_capsules, self.out_capsules, 1)).cuda()
		# b_ij: [in_capsules=1152, out_capsules=10, 1]

		# Iterative routing.
		num_iterations = 3
		for iteration in range(num_iterations):
			# Convert routing logits to softmax.
			c_ij = F.softmax(b_ij.unsqueeze(0))
			c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)
			# c_ij: [batch_size, in_capsules=1152, out_capsules=10, 1, 1]

			# Apply routing `c_ij` to weighted inputs `u_hat`.
			s_j = (c_ij * u_hat).sum(dim=1, keepdim=True) # element-wise product
			# s_j: [batch_size, 1, out_capsules=10, out_capsule_size=16, 1]

			v_j = squash(s_j, dim=3)
			# v_j: [batch_size, 1, out_capsules=10, out_capsule_size=16, 1]

			# Compute inner products of 2 16D-vectors, `u_hat` and `v_j`.
			u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j).squeeze(4).mean(dim=0, keepdim=False)
			# u_vj1: [in_capsules=1152, out_capsules=10, 1]

			# Update b_ij (routing).
			b_ij = b_ij + u_vj1

		return v_j.squeeze(4).squeeze(1) # [batch_size, out_capsules=10, out_capsule_size=16]
