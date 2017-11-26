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
import torchvision.utils as vutils
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
parser.add_argument('--tb-log-interval', type=int, default=10, metavar='N',
					help='how many batches to wait before saving training status to TensorBoard (default: 10)')
parser.add_argument('--tb-image-interval', type=int, default=100, metavar='N',
					help='how many batches to wait before saving reconstructed images to TensorBoard (default: 100)')
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
writer.add_image('original', vutils.make_grid(reconstruction_samples, normalize=True))
reconstruction_samples = Variable(reconstruction_samples, volatile=True).cuda()


# Function to reconstruct the test images
def reconstruct_test_images():
	model.eval()

	output = model(reconstruction_samples)

	reconstructed = model.reconstruct(output)
	reconstructed = reconstructed.data.cpu()

	return reconstructed


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
		loss, margin_loss, reconstruction_loss = model.loss(data, output, target)
		loss.backward()
		optimizer.step()

		print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
			epoch, batch_idx * len(data), len(train_loader.dataset),
			100. * batch_idx / len(train_loader), loss.data[0] )
		)

		reconstructed = reconstruct_test_images()
		vutils.save_image(reconstructed, args.rec_path, normalize=True)
		
		if batch_idx % args.tb_log_interval == 0:
			# Log train/loss to TensorBoard
			n_iter = (epoch - 1) * len(train_loader) + batch_idx + 1
			writer.add_scalar('train/loss', loss.data[0], n_iter)
			writer.add_scalar('train/margin_loss', margin_loss.data[0], n_iter)
			writer.add_scalar('train/reconstruction_loss', reconstruction_loss.data[0], n_iter)

		if batch_idx % args.tb_image_interval == 0:
			# Log reconstructed test images to TensorBoard
			writer.add_image(
				'reconstructed/iter_{}'.format(n_iter), 
				vutils.make_grid(reconstructed, normalize=True)
			)


# Function for testing.
def test(epoch):
	model.eval()
	test_loss, test_margin_loss, test_rec_loss = 0, 0, 0
	correct = 0

	for data, target in test_loader:
		target_indices = target
		target_one_hot = to_one_hot(target_indices)
		data, target = Variable(data, volatile=True).cuda(), Variable(target_one_hot).cuda()

		output = model(data)

		# Sum up batch loss by `size_average=False`, later being averaged over all test samples.
		loss, margin_loss, reconstruction_loss = model.loss(data, output, target, size_average=False)
		loss, margin_loss, reconstruction_loss = loss.data[0], margin_loss.data[0], reconstruction_loss.data[0]

		test_loss += loss
		test_margin_loss += margin_loss
		test_rec_loss += reconstruction_loss
		
		v_mag = torch.sqrt((output**2).sum(dim=2, keepdim=True))
		pred = v_mag.data.max(1, keepdim=True)[1].cpu()
		correct += pred.eq(target_indices.view_as(pred)).sum()

	# Average over all test samples.
	test_loss /= len(test_loader.dataset)
	test_margin_loss /= len(test_loader.dataset)
	test_rec_loss /= len(test_loader.dataset)

	test_accuracy = 100. * correct / len(test_loader.dataset)
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset), test_accuracy )
	)

	# Log test/loss and test/accuracy to TensorBoard at every epoch
	n_iter = epoch * len(train_loader)
	writer.add_scalar('test/loss', test_loss, n_iter)
	writer.add_scalar('test/margin_loss', test_margin_loss, n_iter)
	writer.add_scalar('test/reconstruction_loss', test_rec_loss, n_iter)
	writer.add_scalar('test/accuracy', test_accuracy, n_iter)


# Start training.
for epoch in range(1, args.epochs + 1):
	train(epoch)
	test(epoch)

# Close TensorBoardX summary writer
writer.close()
