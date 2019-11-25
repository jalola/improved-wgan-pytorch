# Improved Training of Wasserstein GANs in Pytorch

This is a Pytorch implementation of [`gan_64x64.py`](https://github.com/igul222/improved_wgan_training/blob/master/gan_64x64.py) from [Improved Training of Wasserstein GANs](https://github.com/igul222/improved_wgan_training).

# To do:
- [x] Support parameters in cli *
- [x] Add requirements.txt *
- [ ] Add Dockerfile if possible
- [x] Multiple GPUs *
- [x] Clean up code, remove unused code *

**\*** not ready for conditional gan yet

# Run

* Example:

**Fresh training**
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --train_dir /path/to/train --validation_dir /path/to/validation/ --output_path /path/to/output/ --dim 64 --saving_step 300 --num_workers 8
```

**Continued training:**
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --train_dir /path/to/train --validation_dir /path/to/validation/ --output_path /path/to/output/ --dim 64 --saving_step 300 --num_workers 8 --restore_mode --start_iter 5000
```

# Model

* `train.py`: This model is mainly based on `GoodGenerator` and `GoodDiscriminator` of `gan_64x64.py` model from [Improved Training of Wasserstein GANs](https://github.com/igul222/improved_wgan_training). It has been trained on LSUN dataset for around 100k iters.
* `congan_train.py`: ACGAN implementation, trained on 4 classes of LSUN dataset

# Result

## 1. WGAN: trained on bedroom dataset (100k iters)

Sample 1            |  Sample 2
:-------------------------:|:-------------------------:
![](https://github.com/jalola/improved-wgan-pytorch/raw/master/result/samples_1.png)  |  ![](https://github.com/jalola/improved-wgan-pytorch/raw/master/result/samples_2.png)

## 2. ACGAN: trained on 4 classes (100k iters)
* dining_room: 1
* bridge: 2
* restaurant: 3
* tower: 4

Sample 1            |  Sample 2
:-------------------------:|:-------------------------:
![](https://github.com/jalola/improved-wgan-pytorch/raw/master/result/acgan_samples_1.png)  |  ![](https://github.com/jalola/improved-wgan-pytorch/raw/master/result/acgan_samples_2.png)

# Testing
During the implementation of this model, we built a test module to compare the result between original model (Tensorflow) and our model (Pytorch) for every layer we implemented. It is available at [compare-tensorflow-pytorch](https://github.com/jalola/compare-tensorflow-pytorch)

# TensorboardX
Results such as costs, generated images (every 200 iters) for tensorboard will be written to `./runs` folder.

To display the results to tensorboard, run: `tensorboard --logdir runs`

# Acknowledgements

* [igul222/improved_wgan_training](https://github.com/igul222/improved_wgan_training)
* [caogang/wgan-gp](https://github.com/caogang/wgan-gp)
* [LayerNorm](https://github.com/pytorch/pytorch/issues/1959)
