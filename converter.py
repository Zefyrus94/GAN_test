import torch
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms

TRAIN_DATA_PATH = './data'
train_data = datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=None)
batchsize = 8
# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batchsize,
                                           shuffle=True)

for i, (images, labels) in enumerate(train_loader):
    print(i)
exit("esco...")
###

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(3)
    ])

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

import pathlib, PIL, random, os, gzip
from pathlib import Path
import PIL.Image
import numpy as np

def load_mnist(path, kind='train'):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels
  
def save_mnist(path, images, labels):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    # prep 10 dirs
    for l in range(10): (p/str(l)).mkdir(parents=True, exist_ok=True)
    for i, (im,l) in enumerate(zip(images, labels)):
        #print(i, im, l)
        dest = p/str(l)/f"{i}.jpg"
        im = im.reshape(28, 28)
        im = PIL.Image.fromarray(im, mode='L')
        with dest.open(mode='wb') as f: im.save(f)

def split_pct(images, labels, pct=0.8):
    items = len(images)
    idx = list(range(items))
    split = int(items*pct) 
    #print(idx, split)
    random.shuffle(idx)
    train_idx = idx[:split]
    valid_idx = idx[split:]
    return images[train_idx], labels[train_idx], images[valid_idx], labels[valid_idx]

def mnist_to_imagenet_format():
    # convert to imagenet image format
    images, labels = load_mnist('data/raw', 'train')
    # split 80% train / 20% valid
    images_trn, labels_trn, images_val, labels_val = split_pct(images, labels, 0.8)
    save_mnist('data/train', images_trn, labels_trn)
    save_mnist('data/valid', images_val, labels_val)

    # test
    images, labels = load_mnist('data/raw', 't10k')
    save_mnist('data/test', images, labels)
    
mnist_to_imagenet_format()