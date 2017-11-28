# CapsNet-PyTorch

A PyTorch implementation of CapsNet based on Geoffrey Hinton's paper [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829).

![capsVSneuron](images/capsule_vs_neuron.png)

This figure is from [CapsNet-Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow).

## Current Status
- The current `test accuracy =  xx.xx`, see `Results` section for details.
- Trying to find the reason why the test accuracy is lower than the one reported in the paper.

## Requirements

- [PyTorch](http://pytorch.org/) (with CUDA)
- [TensorBoard](https://github.com/tensorflow/tensorboard)
- [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch)

## Usage

**Step 1.** Clone this repository

```bash
$ git clone https://github.com/motokimura/CapsNet-PyTorch.git
$ cd CapsNet-PyTorch
```

**Step 2.** Start the training

```bash
$ python main.py
```

**Step 3.** Check training status and validation accuracy from TensorBoard

```bash
# In another terminal window, 
$ cd CapsNet-PyTorch
$ tensorboard --logdir ./runs

# Then, open "http://localhost:6006" from your browser and 
# you will see something like the screenshots in the `Results` section.
```

Some training hyper parameters can be specified from the command line options of `main.py`. 

In default, batch size is 128 both for training and validation, and epoch is set to 50. 
Learning rate of Adam optimizer is set to 0.001 and is exponentially decayed every epoch with the factor of 0.9. 

For more details, type `python main.py --help`.

## Results

Comming soon..

## References

- [naturomics/CapsNet-Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow)
- [XifengGuo/CapsNet-Keras](https://github.com/XifengGuo/CapsNet-Keras)
- [timomernick/pytorch-capsule](https://github.com/timomernick/pytorch-capsule)
