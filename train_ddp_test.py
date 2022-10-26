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
    return trainloader, testloader

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
    trainloader, testloader = create_data_loader_cifar10()
    #net = torchvision.models.resnet50(False).cuda()

    ## Convert BatchNorm to SyncBatchNorm. 
    #net = nn.SyncBatchNorm.convert_sync_batchnorm(net)

    local_rank = int(os.environ['LOCAL_RANK'])
    #net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank])

    start_train = time.time()
    #train(net, trainloader)
    end_train = time.time()
    # save
    #if is_main_process:
    #    save_on_master(net.state_dict(), PATH)
    dist.barrier()

    ## test
    #test(net, PATH, testloader)

    end = time.time()
    seconds = (end - start)
    seconds_train = (end_train - start_train)
    print(f"Total elapsed time: {seconds:.2f} seconds, \
     Train 1 epoch {seconds_train:.2f} seconds")