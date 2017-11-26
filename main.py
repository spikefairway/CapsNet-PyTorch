#
# Dynamic Routing Between Capsules
# https://arxiv.org/pdf/1710.09829.pdf
#

from __future__ import print_function
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F

from capsule_network import CapsuleNetwork


# Get training parameter settings.

parser = argparse.ArgumentParser(description='CapsNet for MNIST')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
					help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
					help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
					help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
					help='learning rate (default: 0.01)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
					help='random seed (default: 1)')
parser.add_argument('--rec-path', default='reconstruction.png', metavar='R',
					help='path to save reconstructed test images (default: reconstruction.png)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
					help='how many batches to wait before logging training status (default: 1)')
parser.add_argument('--tb-log-interval', type=int, default=1, metavar='N',
					help='how many batches to wait before saving TensorBoard event file (default: 10)')
parser.add_argument('--tb-log-dir', default=None, metavar='LD',
					help='directory to output TensorBoard event file (default: runs/<DATETIME>)')

args = parser.parse_args()


# Setup TensorBoardX summary writer
from tensorboardX import SummaryWriter
from datetime import datetime
import os

if args.tb_log_dir is None:
	args.tb_log_dir = os.path.join('runs', datetime.now().strftime('%b%d_%H-%M-%S'))
writer = SummaryWriter(log_dir=args.tb_log_dir)


# Initialize the random seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


# Setup data loaders for train/test sets
kwargs = {'num_workers': 1, 'pin_memory': True}

transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = torch.utils.data.DataLoader(
	datasets.MNIST(
		'data', 
		train=True, 
		download=True,
		transform=transform
	),
	batch_size=args.batch_size, 
	shuffle=True, 
	**kwargs
)

test_loader = torch.utils.data.DataLoader(
	datasets.MNIST(
		'data', 
		train=False, 
		download=True,
		transform=transform
	),
	batch_size=args.test_batch_size, 
	shuffle=True, 
	**kwargs
)


# Build CapsNet.
model = CapsuleNetwork().cuda()
print(model)


# Setup optimizer.
optimizer = optim.Adam(model.parameters(), lr=args.lr)


# Get some random test images for reconstruction testing
test_iter = iter(test_loader)
reconstruction_samples, _ = test_iter.next()
reconstruction_samples = Variable(reconstruction_samples, volatile=True).cuda()


# Function to reconstruct the test images
def reconstruct_test_images():
	model.eval()

	output = model(reconstruction_samples)
	model.reconstruct(output, save_path=args.rec_path)


# Function to convert batches of class indices to classes of one-hot vectors.
def to_one_hot(x, length=10):
	batch_size = x.size(0)
	x_one_hot = torch.zeros(batch_size, length)
	for i in range(batch_size):
		x_one_hot[i, x[i]] = 1.0
	return x_one_hot


# Function for training.
def train(epoch):
	model.train()

	for batch_idx, (data, target) in enumerate(train_loader):
		target_one_hot = to_one_hot(target)
		data, target = Variable(data).cuda(), Variable(target_one_hot).cuda()

		optimizer.zero_grad()
		output = model(data) # forward
		loss = model.loss(data, output, target)
		loss.backward()
		optimizer.step()

		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.data[0] )
			)

			reconstruct_test_images()

		if batch_idx % args.tb_log_interval == 0:
			# Log train/loss to TensorBoard at every iteration
			n_iter = (epoch - 1) * len(train_loader) + batch_idx + 1
			writer.add_scalar('train/loss', loss.data[0], n_iter)


# Function for testing.
def test(epoch):
	model.eval()
	test_loss = 0
	correct = 0

	for data, target in test_loader:
		target_indices = target
		target_one_hot = to_one_hot(target_indices)
		data, target = Variable(data, volatile=True).cuda(), Variable(target_one_hot).cuda()

		output = model(data)

		# Sum up batch loss by `size_average=False`, later being averaged over all test samples.
		test_loss += model.loss(data, output, target, size_average=False).data[0]
		
		v_mag = torch.sqrt((output**2).sum(dim=2, keepdim=True))
		pred = v_mag.data.max(1, keepdim=True)[1].cpu()
		correct += pred.eq(target_indices.view_as(pred)).sum()

	test_loss /= len(test_loader.dataset) # Average over all test samples.
	test_accuracy = 100. * correct / len(test_loader.dataset)
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset), test_accuracy )
	)

	# Log test/loss and test/accuracy to TensorBoard at every epoch
	n_iter = epoch * len(train_loader)
	writer.add_scalar('test/loss', test_loss, n_iter)
	writer.add_scalar('test/accuracy', test_accuracy, n_iter)


# Visualize network as a graph on TensorBoard
res = model(torch.autograd.Variable(torch.Tensor(1, 1, 28, 28), requires_grad=True))
writer.add_graph(model, lastVar=res)


# Start training.
for epoch in range(1, args.epochs + 1):
	train(epoch)
	test(epoch)

# Close TensorBoardX summary writer
writer.close()