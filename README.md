## Capsules Pytorch

Yet another implementation of [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829) paper in Pytorch.

Disclaimer: **WIP**. Proceed at your own will.

## Requirements

+ Python >= 3.4

+ [Pytorch](http://pytorch.org/) >= 0.3

+ [TorchVision](https://github.com/pytorch/vision)

+ [tensorboardX](https://github.com/lanpa/tensorboard-pytorch)

+ [tqdm](https://github.com/tqdm/tqdm)

## Usage

+ Training: `python main.py`

GPU | Training speed (min/epoch)
:--:|:-------------------------:
1 x GeForce GTX 960 | 11
1 x GeForce GTX 1080Ti | 2

+ Tensorboard: `tensorboard --logdir logs` (check http://localhost:6006)

## Experiment

Dataset | Routing | Reconstruction | Best test error | Best test accuracy
:------:|:---:|:----:|:----:|:------:
Fashion MNIST | 3 | yes | 0.1850 | 88.11%

## References

+ Code base is heavily borrowed from https://github.com/cedrickchee/capsule-net-pytorch
