# Local Mix Implicit Network for Arbitrary-Scale Image Super-Resolution

Official PyTorch implementation of Local Mix Implicit Network for Arbitrary-Scale Image Super-Resolution.

## Installation

Our code is based on Ubuntu 20.04, pytorch 1.11.0, CUDA 11.3 (NVIDIA RTX 3090 24GB, NVIDIA A40 48GB) and python 3.8.

## Note

We have hidden the core code for now, and this part of the code will be made public after the paper is accepted.

## Train & Test

##### **EDSR-baseline-LMI**

**Train**: `python train_real.py --config configs/train-real/lmi-edsr-baseline.yaml`

**Test**: `python test_real.py --config configs/test/test-RealAbrSR.yaml --model save/LMI-edsr-baseline/epoch-best.pth`

##### **RDN-LMI**

**Train**: `python train_real.py --config configs/train-real/lmi-rdn.yaml `

**Test**: `python test_real.py --config configs/test/test-RealAbrSR.yaml --model save/LMI-rdn/epoch-best.pth`

We use NVIDIA RTX 3090 24GB for training, and NVIDIA A40 48GB for testing.

## Citation

If you find our work useful in your research, please consider citing our paper:

## Acknowledgements

This code is built on [LIIF](https://github.com/yinboc/liif) and [LTE](https://github.com/jaewon-lee-b/lte).  We thank the authors for sharing their codes.
