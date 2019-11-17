import os, sys
sys.path.append(os.getcwd())
import click
import time
import functools
import pdb

import numpy as np
from pathlib import Path

import torch
import torchvision
from torch import optim
from torchvision import transforms


from models.wgan import *
from training_utils import *
import libs as lib
import libs.plot
from tensorboardX import SummaryWriter

# to fix png loading
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from timeit import default_timer as timer


# Some public testing data
# lsun lmdb data set can be download via https://github.com/fyu/lsun
# ImageNet (64x64) at http://image-net.org/small/download.php

@click.command()
@click.option('--train_dir', default=None, help='Data path for training')
@click.option('--validation_dir', default=None, help='Data path for valication')
@click.option('--image_data_type', default="image_folder", type=click.Choice(["lsun", "image_folder"]), help='If you are using lsun images from lsun lmdb, use lsun. If you use your own data in a folder, then use "image_folder". If you use lmdb, you\'ll need to write the loader by yourself. Please check load_data function')
@click.option('--output_path', default=None, help='Output path where result (.e.g drawing images, cost, chart) will be stored')
@click.option('--dim', default=64, help='Model dimensionality or image resolution, tested with 64.')
@click.option('--lr', default=1e-4, help='Learning rate')
@click.option('--critic_iters', default=5, help='How many iterations to train the critic/disciminator for')
@click.option('--gen_iters', default=1, help='How many iterations to train the gemerator for')
@click.option('--batch_size', default=64, help='Training batch size. Must be a multiple of number of gpus')
@click.option('--noisy_label_prob', default=0., help='Make the labels the noisy for the discriminator: occasionally flip the labels when training the discriminator')
@click.option('--start_iter', default=0, help='Starting iteration')
@click.option('--end_iter', default=100000, help='Ending iteration')
@click.option('--gp_lambda', default=10, help='Gradient penalty lambda hyperparameter')
@click.option('--num_workers', default=5, help='Number of workers to load data')
@click.option('--saving_step', default=200, help='Save model, sample every this saving step')
@click.option('--training_class', default=None, help='A list of classes, separated by comma ",". IGNORE this if you are NOT training on lsun, or if you want to train on other classes of lsun, then change it accordingly')
@click.option('--val_class', default=None, help='A list of classes, separated by comma ",". IGNORE this if you are NOT training on lsun, or if you want to train on other classes of lsun, then change it accordingly')
@click.option('--restore_mode/--no-restore_mode', default=False, help="If True, it will load saved model from OUT_PATH and continue to train")


def train(train_dir, validation_dir, image_data_type, output_path, dim, lr, critic_iters, gen_iters, batch_size, noisy_label_prob, start_iter, end_iter, gp_lambda, num_workers, saving_step, training_class, val_class, restore_mode):

    if train_dir is None or len(train_dir) == 0:
        raise Exception('Please specify path to data directory in gan.py!')


    output_path = Path(output_path)
    sample_path = output_path / "samples"
    mkdir_path(sample_path)
    if isinstance(training_class, str):
        training_class = training_class.split(",")
    if isinstance(val_class, str):
        val_class = val_class.split(",")

    data_transform = transforms.Compose([
        transforms.Resize(dim),
        transforms.RandomCrop(dim),
        # transforms.Lambda(lambda x : x + torch.normal(0, 0.1, (3, dim, dim))),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x : x + torch.randn_like(x)),
        # transforms.Lambda(lambda x : x + torch.normal(0, 0.1, (3, dim, dim))),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
    ])

    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    fixed_noise = gen_rand_noise(batch_size).to(device)

    if restore_mode:
        aG = GoodGenerator(dim, dim*dim*3)
        aD = GoodDiscriminator(dim)
        # aG = torch.load(str(output_path / "generator.pt"))
        # aD = torch.load(str(output_path / "discriminator.pt"))
        g_state_dict = torch.load(str(output_path / "generator.pt"))
        aG.load_state_dict(remove_module_str_in_state_dict(g_state_dict))
        d_state_dict = torch.load(str(output_path / "discriminator.pt"))
        aD.load_state_dict(remove_module_str_in_state_dict(d_state_dict))
    else:
        aG = GoodGenerator(dim, dim*dim*3)
        aD = GoodDiscriminator(dim)
        aG.apply(weights_init)
        aD.apply(weights_init)

    optimizer_g = torch.optim.Adam(aG.parameters(), lr=lr, betas=(0,0.9))
    optimizer_d = torch.optim.Adam(aD.parameters(), lr=lr, betas=(0,0.9))
    one = torch.FloatTensor([1])
    mone = one * -1
    # aG = aG.to(device)
    # aD = aD.to(device)
    aG = torch.nn.DataParallel(aG).to(device)
    aD = torch.nn.DataParallel(aD).to(device)
    one = one.to(device)
    mone = mone.to(device)

    writer = SummaryWriter()
    #Reference: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
    dataloader = load_data(image_data_type, train_dir, data_transform, batch_size=batch_size, classes=training_class, num_workers=num_workers)
    dataiter = iter(dataloader)
    for iteration in range(start_iter, end_iter):
        start_time = time.time()
        print("Iter: " + str(iteration))
        start = timer()
        #---------------------TRAIN G------------------------
        for p in aD.parameters():
            p.requires_grad_(False)  # freeze D

        gen_cost = None
        for i in range(gen_iters):
            print("Generator iters: " + str(i))
            aG.zero_grad()
            noise = gen_rand_noise(batch_size).to(device)
            noise.requires_grad_(True)
            fake_data = aG(noise)
            # fake_data = fake_data.view(batch_size, 3, dim, dim)
            # fake_data += torch.normal(0, 0.1, (batch_size, 3, dim, dim)).to(device)
            gen_cost = aD(fake_data)
            gen_cost = gen_cost.mean()
            gen_cost.backward(mone)
            gen_cost = -gen_cost
        
        optimizer_g.step()
        end = timer()
        print(f'---train G elapsed time: {end - start}')
        #---------------------TRAIN D------------------------
        for p in aD.parameters():  # reset requires_grad
            p.requires_grad_(True)  # they are set to False below in training G
        for i in range(critic_iters):
            print("Critic iter: " + str(i))
            
            start = timer()
            aD.zero_grad()

            # gen fake data and load real data
            noise = gen_rand_noise(batch_size).to(device)
            with torch.no_grad():
                noisev = noise  # totally freeze G, training D
            fake_data = aG(noisev).detach()
            # fake_data = fake_data.view(batch_size, 3, dim, dim)
            # fake_data += torch.normal(0, 0.1, (batch_size, 3, dim, dim)).to(device)
            end = timer(); print(f'---gen G elapsed time: {end-start}')
            start = timer()
            batch = next(dataiter, None)
            if batch is None:
                dataiter = iter(dataloader)
                batch = dataiter.next()
            batch = batch[0] #batch[1] contains labels
            real_data = batch.to(device) #TODO: modify load_data for each loading
            end = timer(); print(f'---load real imgs elapsed time: {end-start}')
            start = timer()

            is_flipping = False
            if noisy_label_prob > 0 and noisy_label_prob < 1:
                is_flipping = np.random.randint(1//noisy_label_prob, size=1)[0] == 1

            if not is_flipping:
                # train with real data
                disc_real = aD(real_data)
                disc_real = disc_real.mean()

                # train with fake data
                disc_fake = aD(fake_data)
                disc_fake = disc_fake.mean()
            else:
                # train with fake data
                disc_real = aD(fake_data)
                disc_real = disc_real.mean()

                # train with real data
                disc_fake = aD(real_data)
                disc_fake = disc_fake.mean()

            #showMemoryUsage(0)
            # train with interpolates data
            gradient_penalty = calc_gradient_penalty(aD, real_data, fake_data, batch_size, dim, device, gp_lambda)
            #showMemoryUsage(0)

            # final disc cost
            disc_cost = disc_fake - disc_real + gradient_penalty
            disc_cost.backward()
            w_dist = disc_fake  - disc_real
            optimizer_d.step()
            #------------------VISUALIZATION----------
            if i == critic_iters-1:
                writer.add_scalar('data/disc_cost', disc_cost, iteration)
                #writer.add_scalar('data/disc_fake', disc_fake, iteration)
                #writer.add_scalar('data/disc_real', disc_real, iteration)
                writer.add_scalar('data/gradient_pen', gradient_penalty, iteration)

            end = timer(); print(f'---train D elapsed time: {end-start}')
        #---------------VISUALIZATION---------------------
        writer.add_scalar('data/gen_cost', gen_cost, iteration)

        lib.plot.plot(str(output_path / 'time'), time.time() - start_time)
        lib.plot.plot(str(output_path / 'train_disc_cost'), disc_cost.cpu().data.numpy())
        lib.plot.plot(str(output_path / 'train_gen_cost'), gen_cost.cpu().data.numpy())
        lib.plot.plot(str(output_path / 'wasserstein_distance'), w_dist.cpu().data.numpy())
        if iteration > 0 and iteration % saving_step == 0:
            val_loader = load_data(image_data_type, validation_dir, data_transform, batch_size=batch_size, classes=val_class, num_workers=num_workers)
            dev_disc_costs = []
            for _, images in enumerate(val_loader):
                imgs = torch.Tensor(images[0])
               	imgs = imgs.to(device)
                with torch.no_grad():
            	    imgs_v = imgs

                D = aD(imgs_v)
                _dev_disc_cost = -D.mean().cpu().data.numpy()
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot(str(output_path / 'dev_disc_cost.png'), np.mean(dev_disc_costs))
            lib.plot.flush()	
            gen_images = generate_image(aG, dim=dim, batch_size=batch_size, noise=fixed_noise)
            torchvision.utils.save_image(gen_images, str(sample_path / 'samples_{}.png').format(iteration), nrow=8, padding=2)
            grid_images = torchvision.utils.make_grid(gen_images, nrow=8, padding=2)
            writer.add_image('images', grid_images, iteration)
	#----------------------Save model----------------------
            # torch.save(aG, str(output_path / "generator.pt"))
            # torch.save(aD, str(output_path / "discriminator.pt"))
            torch.save(aG.state_dict(), str(output_path / "generator.pt"))
            torch.save(aD.state_dict(), str(output_path / "discriminator.pt"))
        lib.plot.tick()

if __name__ == '__main__':
    train()
    