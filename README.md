# Improved Training of Wasserstein GANs in Pytorch

This is a replication of [`gan_64x64.py`](https://github.com/igul222/improved_wgan_training/blob/master/gan_64x64.py) from [Improved Training of Wasserstein GANs](https://github.com/igul222/improved_wgan_training) but written in Pytorch code instead of Tensorflow.

# Prerequisites
* Python >= 3.6
* Pytorch [Latest version from master branch](https://github.com/pytorch/pytorch)
* Numpy
* SciPy
* tensorboardX ([installation here](https://github.com/lanpa/tensorboard-pytorch))

# Models

* `gan_64x64.py`: This model is mainly based on `GoodGenerator` and `GoodDiscriminator` of `gan_64x64.py` model from [Improved Training of Wasserstein GANs](https://github.com/igul222/improved_wgan_training). It has been trained on lsun dataset for around 100k iters.

# Result

Some samples after 100k iters of training

![sample 1](https://github.com/jalola/improved-wgan-pytorch/samples_1.png "Sample 1")
