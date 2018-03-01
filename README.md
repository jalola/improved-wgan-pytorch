# Improved Training of Wasserstein GANs in Pytorch

This is a replication of [`gan_64x64.py`](https://github.com/igul222/improved_wgan_training/blob/master/gan_64x64.py) from [Improved Training of Wasserstein GANs](https://github.com/igul222/improved_wgan_training) but written in Pytorch instead of Tensorflow.

# Prerequisites
* Python >= 3.6
* Pytorch [Latest version from master branch](https://github.com/pytorch/pytorch)
* Numpy
* SciPy
* tensorboardX ([installation here](https://github.com/lanpa/tensorboard-pytorch)). It is very convenient to see costs and results during training with TensorboardX for Pytorch
* TensorFlow for tensorboardX

# Models

* `gan_64x64.py`: This model is mainly based on `GoodGenerator` and `GoodDiscriminator` of `gan_64x64.py` model from [Improved Training of Wasserstein GANs](https://github.com/igul222/improved_wgan_training). It has been trained on LSUN dataset for around 100k iters.

# Result

Some samples after 100k iters of training

Sample 1            |  Sample 2
:-------------------------:|:-------------------------:
![](https://github.com/jalola/improved-wgan-pytorch/raw/master/result/samples_1.png)  |  ![](https://github.com/jalola/improved-wgan-pytorch/raw/master/result/samples_2.png)

# Testing model
During the implementation of this model, we built a test module to compare the result between original model (Tensorflow) and our model (Pytorch). We will make it public soon

# Acknowledge

* [igul222/improved_wgan_training](https://github.com/igul222/improved_wgan_training)
* [caogang/wgan-gp](https://github.com/caogang/wgan-gp)
* [LayerNorm](https://github.com/pytorch/pytorch/issues/1959)