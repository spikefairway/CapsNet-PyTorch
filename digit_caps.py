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
	def __init__(self, routing_iters, gpu):
		super(DigitCaps, self).__init__()

		self.routing_iters = routing_iters
		self.gpu = gpu

		self.in_capsules = 1152
		self.in_capsule_size = 8
		self.out_capsules = 10
		self.out_capsule_size = 16

		self.W = nn.Parameter(
			torch.randn(
				self.in_capsules, 
				self.in_capsule_size,
				self.out_capsules * self.out_capsule_size 
			)
		)
		# W: [in_capsules, in_capsule_size, out_capsules * out_capsule_size] = [1152, 8, 10*16=160]

		self.b_ij = nn.Parameter(
			torch.zeros((
				self.in_capsules,
				self.out_capsules
			))
		)
		# b_ij: [in_capsules, out_capsules] = [1152, 10]


	# FIXME, write in an easier way to understand, some tensors have some redundant dimensions.
	def forward(self, x):
		# x: [batch_size, in_capsules=1152, in_capsule_size=8]
		batch_size = x.size(0)

		x = x.unsqueeze(2)
		# x: [batch_size, in_capsules=1152, 1, in_capsule_size=8]

		# Transform inputs by weight matrix `W`.
		u_hat = x.matmul(self.W)
		u_hat = u_hat.view(u_hat.size(0), self.in_capsules, self.out_capsules, self.out_capsule_size)
		# u_hat: [batch_size, in_capsules=1152, 10, 16]

		u_hat_detached = u_hat.detach()
		# In forward pass, `u_hat_detached` = `u_hat`, and 
	# in backward, no gradient can flow from `u_hat_detached` back to `u_hat`.

		c_ij = F.softmax(self.b_ij, dim=1)
		# c_ij: [1152, 10]

		s_j = (c_ij.unsqueeze(2) * u_hat_detached).sum(dim=1)
		# s_j: [batch_size, 10, 16]

		v_j = squash(s_j, dim=2)
		# v_j: [batch_size, 10, 16]

		# Iterative routing.
		if self.routing_iters > 0:
			b_batch = self.b_ij.expand((batch_size, self.in_capsules, self.out_capsules))
			if self.gpu > 0:
				b_batch = b_batch.cuda(self.gpu)
			# b_batch: [batch_size, 1152, 10]
			for iteration in range(self.routing_iters):
				v_j = v_j.unsqueeze(1)
				# v_j: [batch_size, 1, 10, 16]
				
				# Update b_ij
				b_batch = b_batch + (u_hat_detached * v_j).sum(-1)
				# (u_hat * v_j).sum(-1) : [batch_size, 1152, 10]

				c_ij = F.softmax(b_batch.view(-1, self.out_capsules), dim=1).view(-1, self.in_capsules, self.out_capsules, 1)
				# c_ij: [batch_size, 1152, 10, 1]

				if iteration == self.routing_iters - 1:
					# Apply routing `c_ij` to weighted inputs `u_hat`.
					s_j = (c_ij * u_hat).sum(dim=1) # element-wise product
				else:
					s_j = (c_ij * u_hat_detached).sum(dim=1)
				# s_j: [batch_size, out_capsules=10, out_capsule_size=16]
	
				v_j = squash(s_j, dim=2)
				# v_j: [batch_size, out_capsules=10, out_capsule_size=16]

		return v_j # [batch_size, out_capsules=10, out_capsule_size=16]
