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
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-decay-factor', type=float, default=0.9, metavar='DF',
                    help='factor to decay learning rate (default: 0.9)')
parser.add_argument('--lr-decay-epoch', type=int, default=1, metavar='DE',
                    help='how many epochs to wait before decaying learning rate (default: 1)')
parser.add_argument('--routing', type=int, default=3, metavar='R',
                    help='iteration numbers for dymanic routing b/w capsules (default: 3)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--org-path', default='original.png', metavar='O',
                    help='path to save test images to reconstruct (default: original.png)')
parser.add_argument('--rec-path', default='reconstructed.png', metavar='R',
                    help='path to save reconstructed test images (default: reconstructed.png)')
parser.add_argument('--tb-log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before saving training status to TensorBoard (default: 10)')
parser.add_argument('--tb-image-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before saving reconstructed images to TensorBoard (default: 100)')
parser.add_argument('--tb-log-dir', default=None, metavar='LD',
                    help='directory to output TensorBoard event file (default: runs/<DATETIME>)')
parser.add_argument('--gpu', type=int, default=0, metavar='G',
                    help='id of the GPU to use (default: 0)')

args = parser.parse_args()


# Check CUDA availability.
if args.gpu >= 0:
    assert torch.cuda.is_available(), \
        'Aborted. CUDA seems to be not available. Use `--gpu -1` option to train with CPUs.'


# Setup TensorBoardX summary writer.
from tensorboardX import SummaryWriter
from datetime import datetime
import os
from tqdm import tqdm

if args.tb_log_dir is None:
    args.tb_log_dir = os.path.join('runs', datetime.now().strftime('%b%d_%H-%M-%S'))
writer = SummaryWriter(log_dir=args.tb_log_dir)


# Initialize the random seed.
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


# Setup data loaders for train/test data.
train_dataset = datasets.MNIST(
    'data', train=True, download=True, 
    transform=transforms.Compose([
        transforms.RandomCrop(padding=2, size=(28, 28)), # data augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
)

test_dataset = datasets.MNIST(
    'data', train=False, download=True, 
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
)

kwargs = {'num_workers': 1, 'pin_memory': True} if (args.gpu >= 0) else {}

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size, 
    shuffle=True, 
    **kwargs
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.test_batch_size, 
    shuffle=True, 
    **kwargs
)


# Build CapsNet.
model = CapsuleNetwork(routing_iters=args.routing, gpu=args.gpu)
if args.gpu >=0:
    model = model.cuda(args.gpu)

print(model)


# Setup optimizer.
optimizer = optim.Adam(model.parameters(), lr=args.lr)


# Get some random test images for reconstruction testing.
test_iter = iter(test_loader)
reconstruction_samples, _ = test_iter.next()

vutils.save_image(reconstruction_samples, args.org_path, normalize=True)
writer.add_image('original', vutils.make_grid(reconstruction_samples, normalize=True))

reconstruction_samples = Variable(reconstruction_samples, volatile=True)
if args.gpu >= 0:
    reconstruction_samples = reconstruction_samples.cuda(args.gpu)


# Function to reconstruct the test images.
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


# Function to get learning rates from the optimizer.
def get_lr():
    lr_params = []
    for param_group in optimizer.param_groups:
        lr_params.append(param_group['lr'])
    return lr_params


# Function to decay learning rate.
def decay_lr(epoch):
    if epoch % args.lr_decay_epoch != (args.lr_decay_epoch - 1):
        return
    for param_group in optimizer.param_groups:
        param_group['lr'] *= args.lr_decay_factor

# Function to calculate accuracy
def accuracy(target, output):
    """
        target: [batch_size]
        output: [batch_size, num_digit_capsule=10, digit_capsule_size=16]
    """
    v_mag = torch.sqrt((output ** 2).sum(dim=2, keepdim=True))
    # v_mag: [batch_size, num_digit_capsule=10, 1]

    pred = v_mag.data.max(1, keepdim=True)[1].cpu()
    # pred: [batch_size, 1, 1]
    correct_pred = pred.eq(target.view_as(pred))

    return correct_pred.float().mean()

# Function for training.
def train(epoch):
    model.train()

    total_acc = 0.
    total_n_iter = args.epochs * len(train_loader)

    for batch_idx, (data, target) in enumerate(tqdm(train_loader, unit='batch')):
        target_indices = target
        target_one_hot = to_one_hot(target)

        data, target = Variable(data), Variable(target_one_hot)
        if args.gpu >= 0:
            data, target = data.cuda(args.gpu), target.cuda(args.gpu)

        optimizer.zero_grad()
        output = model(data) # forward.
        loss, margin_loss, reconstruction_loss = model.loss(data, output, target)
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        acc = accuracy(target_indices, output)
        total_acc += acc
        avg_acc = total_acc / (batch_idx + 1)

        """
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * args.batch_size, len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.data[0] )
        )
        """

        reconstructed = reconstruct_test_images()
        vutils.save_image(reconstructed, args.rec_path, normalize=True)
        
        n_iter = epoch * len(train_loader) + batch_idx

        template = '\rEpoch {}, ' \
                'Step {}/{}: ' \
                '[Total loss: {:.6f},' \
                '\tMargin loss: {:.6f},' \
                '\tReconstruction loss: {:.6f},' \
                '\tBatch accuracy: {:.6f},' \
                '\tAccuracy: {:.6f}]'
        tqdm.write(template.format(
            epoch,
            n_iter,
            total_n_iter,
            loss.data[0],
            margin_loss.data[0],
            reconstruction_loss.data[0],
            acc,
            avg_acc))


        if n_iter % args.tb_log_interval == 0:
            # Log train/loss to TensorBoard.
            writer.add_scalar('train/loss', loss.data[0], n_iter)
            writer.add_scalar('train/loss_margin', margin_loss.data[0], n_iter)
            writer.add_scalar('train/loss_reconstruction', reconstruction_loss.data[0], n_iter)
            writer.add_scalar('train/batch_accuracy', acc, n_iter)
            writer.add_scalar('train/accuracy', avg_acc, n_iter)

            # Log base learning rate to TensorBoard.
            lr = get_lr()[0]
            writer.add_scalar('lr', lr, n_iter)

        if n_iter % args.tb_image_interval == 0:
            # Log reconstructed test images to TensorBoard.
            writer.add_image(
                'reconstructed/iter_{}'.format(n_iter), 
                vutils.make_grid(reconstructed, normalize=True)
            )

    decay_lr(epoch)


# Function for testing.
def test(epoch):
    model.eval()
    test_loss, test_margin_loss, test_rec_loss = 0, 0, 0
    total_acc = 0

    for data, target in test_loader:
        target_indices = target
        target_one_hot = to_one_hot(target_indices)

        data, target = Variable(data, volatile=True), Variable(target_one_hot)
        if args.gpu >= 0:
            data, target = data.cuda(args.gpu), target.cuda(args.gpu)

        output = model(data)

        # Sum up batch loss by `size_average=False`, later being averaged over all test samples.
        loss, margin_loss, reconstruction_loss = model.loss(data, output, target, size_average=False)
        loss, margin_loss, reconstruction_loss = loss.data[0], margin_loss.data[0], reconstruction_loss.data[0]

        test_loss += loss
        test_margin_loss += margin_loss
        test_rec_loss += reconstruction_loss
        
        acc = accuracy(target_indices, output)
        total_acc += acc

    # Average over all test samples.
    test_loss /= len(test_loader.dataset)
    test_margin_loss /= len(test_loader.dataset)
    test_rec_loss /= len(test_loader.dataset)
    test_acc = total_acc / len(test_loader)

    print('\nEpoch {}: Test average loss: {:.4f}, Test accuracy: {:.4f}\n'.format(
        epoch, test_loss, test_acc )
    )

    # Log test/loss and test/accuracy to TensorBoard at every epoch.
    n_iter = epoch * len(train_loader)
    writer.add_scalar('test/loss', test_loss, n_iter)
    writer.add_scalar('test/loss_margin', test_margin_loss, n_iter)
    writer.add_scalar('test/loss_reconstruction', test_rec_loss, n_iter)
    writer.add_scalar('test/accuracy', test_acc, n_iter)


# Start training.
for epoch in range(args.epochs):
    train(epoch)
    test(epoch)

# Close TensorBoardX summary writer.
writer.close()
