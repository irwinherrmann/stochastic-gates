# Channel-selection using Gumbel Softmax

This repository contains a PyTorch implementation of the paper  "Channel-selection using Gumbel Softmax" (previously titled "An end-to-end approach for speeding up neural networks"). Published at ECCV '20.

ECCV link: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720239.pdf
Arxiv link: https://arxiv.org/abs/1812.04180

Note that this code results in a pruned network with higher accuracy than the baseline at rough half of the FLOPs (ResNet 50: 76.14 with only 55% of original FLOPs).

# Requirements

This implementation is developed for

PyTorch 0.3.1
CUDA 9.1

Code uses PyTorch 0.3.1. We timed it and found that this code is faster in 0.3.1 than 0.4.1. Have not tested it with 1+ yet.

Note, this code very closely follows the code for Convolutional Networks with Adaptive Inference Graphs (ConvNet-AIG): https://github.com/andreasveit/convnet-aig. Thanks to Andreas Veit and Serge Belongie for all the help.

# Citing

If you find this helps your research, please consider citing:

@inproceedings{herrmann2020channel,
  title={Channel Selection Using Gumbel Softmax},
  author={Herrmann, Charles and Bowen, Richard Strong and Zabih, Ramin},
  booktitle={European Conference on Computer Vision},
  pages={241--257},
  year={2020},
  organization={Springer}
}
