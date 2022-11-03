import os 
import math
from torchsummary import summary
from torchvision.models import inception_v3
from torch.distributions import MultivariateNormal
import seaborn as sns
import scipy
import numpy as np
import pandas as pd
import imageio
from custom_celeba import CelebACustom
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm.auto import tqdm
import os
from os import listdir
from os.path import isfile, join
import click
import torch.nn.functional as F
from importlib import import_module#dynamic import
import zipfile
##
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import time
import torchvision
from utils import setup_for_distributed, save_on_master, is_main_process


gpu_list = "0,1,2,3"
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
n_classes = 10
data_shape = (1, 28, 28)
image_width = 64
image_height = 64
image_size = 299
image_channels = 3
image_path = None
ckpt_path = None
history_path = None
gif_path = None
start_epoch = 0
n_epochs = 100
batch_size = 128
z_dim = 64
display_step = 400
batch_size = 2
lr = 0.0002
beta_1 = 0.5 
beta_2 = 0.999
device = 'cuda'
c_lambda = 10
crit_repeats = 5
gen = None
disc = None
gen_opt = None
disc_opt = None
criterion = None
dataloader = None

def create_data_loader_cifar10():
    transform = transforms.Compose(
        [
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 256

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)                                  
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=trainset, shuffle=True)                                                  
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            sampler=train_sampler, num_workers=16, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    test_sampler = torch.utils.data.distributed.DistributedSampler(dataset=testset, shuffle=True)                                         
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, sampler=test_sampler, num_workers=16)
    return trainloader#, testloader
#ut
def get_noise(n_samples, z_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    '''
    return torch.randn(n_samples, z_dim, device=device)
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
#cgan
def get_one_hot_labels(labels, n_classes):
    return F.one_hot(labels, n_classes)
def combine_vectors(x, y):
    #print(x.shape,y.shape)
    #torch.Size([2, 3, 64, 64]) torch.Size([2, 10, 32, 32])
    #Sizes of tensors must match except in dimension 2.
    #Got 32 and 64 (The offending index is 0)
    combined = torch.cat((x.float(), y.float()), 1)
    return combined
def get_input_dimensions(z_dim, data_shape, n_classes):
    generator_input_dim = z_dim+n_classes
    discriminator_im_chan = data_shape[0]+n_classes
    return generator_input_dim, discriminator_im_chan
#
def train(gen, disc, trainloader):
    print("Start training...")
    criterion = nn.BCEWithLogitsLoss()
    gen_input_dim = z_dim
    disc_input_dim = image_channels
    gen_input_dim, disc_input_dim = get_input_dimensions(z_dim, data_shape, n_classes)
    print("disc_input_dim",disc_input_dim)
    gen = Generator(input_dim=gen_input_dim).to(device)
    disc = Discriminator(disc_input_dim).to(device) 
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
    print("Architettura Generatore")
    #summary(gen, insize_g)
    #print("gen_input_dim",gen_input_dim)
    summary(gen, (gen_input_dim,))
    print("Architettura Discriminatore")
    #print("disc_input_dim",disc_input_dim)
    #summary(disc, insize_d)
    summary(disc, (disc_input_dim,image_width,image_height))  
    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)
    epochs = 1
    num_of_batches = len(trainloader)
    for epoch in range(start_epoch, epochs):  # loop over the dataset multiple times
        trainloader.sampler.set_epoch(epoch)
        running_loss_d = 0.0
        running_loss_g = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the real; data is a list of [real, labels]
            real, labels = data
            #real, labels = real.cuda(), labels.cuda() 
            # zero the parameter gradients
            real = real.to(device)
            labels = labels.to(device)
            one_hot_labels = get_one_hot_labels(labels, n_classes)
            image_one_hot_labels = one_hot_labels[:, :, None, None]
            image_one_hot_labels = image_one_hot_labels.repeat(1, 1, data_shape[1], data_shape[2])
            prev_fake = None#per differenza con la cgan
            prev_real = real#per differenza con la cgan
            ####
            disc_opt.zero_grad()
            fake_noise = get_noise(cur_batch_size, z_dim, device=device)
            fake_noise = combine_vectors(fake_noise,one_hot_labels)
            fake = gen(fake_noise)
            #print("fake shape",fake.shape)
            prev_fake = fake#per differenza con la cgan
            fake_image_and_labels = combine_vectors(fake,image_one_hot_labels)
            #print("fake1 shape",fake.shape)
            real_image_and_labels = combine_vectors(real,image_one_hot_labels)
            disc_fake_pred = disc(fake_image_and_labels)
            disc_real_pred = disc(real_image_and_labels)
            disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            #print("rshape",real.shape)
            disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            running_loss_d += disc_loss.item()
            # Keep track of the average discriminator loss    
            mean_discriminator_loss += disc_loss.item() / display_step    
            # Update gradients
            disc_loss.backward(retain_graph=True)
            # Update optimizer
            disc_opt.step()
            ## Update generator ##
            gen_opt.zero_grad()
            fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)
            disc_fake_pred = disc(fake_image_and_labels)
            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
            running_loss_g += gen_loss.item()
            gen_loss.backward()
            gen_opt.step()
            mean_generator_loss += gen_loss.item() / display_step
            """
            if cur_step % display_step == 0 and cur_step > 0:
                print(f"Epoch {epoch} Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
                #show_tensor_images(fake,name=epoch,gen=gen,disc=disc)
                print("plot ",prev_fake.shape)
                show_tensor_images(net, prev_fake, epoch_num=epoch, gen=gen, disc=disc, gen_opt=gen_opt, disc_opt=disc_opt)
                save_step += 1
                show_tensor_images(net, prev_real)
                mean_generator_loss = 0
                mean_discriminator_loss = 0
            """
            cur_step += 1 
            ####
            """
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(real)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            """
        print(f'[Epoch {epoch + 1}/{epochs}] loss d: {running_loss_d / num_of_batches:.3f}; loss g: {running_loss_g / num_of_batches:.3f}')
    
    print('Finished Training')

def init_distributed():

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)
    # synchronizes all the threads to reach this point before moving on
    dist.barrier()
    setup_for_distributed(rank == 0)

if __name__ == '__main__':
    start = time.time()
    
    init_distributed()
    
    PATH = './cifar_net.pth'
    PATH_D = './cifar_disc.pth'
    PATH_G = './cifar_gen.pth'
    trainloader = create_data_loader_cifar10()
    #net = torchvision.models.resnet50(False).cuda()
    net = 'cgan'
    from models.cgan.networks.Generator import Generator
    from models.cgan.networks.Discriminator import Discriminator
    gen = Generator.Generator
    disc = Discriminator.Discriminator
    
    ## Convert BatchNorm to SyncBatchNorm. 
    #net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    gen = nn.SyncBatchNorm.convert_sync_batchnorm(gen)
    disc = nn.SyncBatchNorm.convert_sync_batchnorm(disc)
    local_rank = int(os.environ['LOCAL_RANK'])
    #net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank])
    gen = nn.parallel.DistributedDataParallel(gen, device_ids=[local_rank])
    disc = nn.parallel.DistributedDataParallel(disc, device_ids=[local_rank])
    start_train = time.time()
    train(gen, disc, trainloader)
    end_train = time.time()
    # save
    if is_main_process:
        save_on_master(gen.state_dict(), PATH_G)
        save_on_master(disc.state_dict(), PATH_D)
    dist.barrier()

    ## test
    #test(net, PATH, testloader)

    end = time.time()
    seconds = (end - start)
    seconds_train = (end_train - start_train)
    print(f"Total elapsed time: {seconds:.2f} seconds, \
     Train 1 epoch {seconds_train:.2f} seconds")