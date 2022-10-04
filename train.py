#import csv
#floor
import math
#summary
from torchsummary import summary
##inception e fid
from torchvision.models import inception_v3
from torch.distributions import MultivariateNormal
import seaborn as sns
import scipy
import numpy as np
import pandas as pd
#gif
import imageio
#importare celebacustom
#
from custom_celeba import CelebACustom
#gen e disc
#data loading e transform
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
#criterion...
import torch
from torch import nn
from tqdm.auto import tqdm
####
import os
from os import listdir
from os.path import isfile, join
#
import click
#cgan
import torch.nn.functional as F
#
from importlib import import_module#dynamic import
#
n_classes = 10#per cgan CIFAR-10, MNIST
data_shape = (1, 28, 28)#per cgan CIFAR-10, MNIST
image_width = 64#299
image_height = 64#299
image_size = 299
image_channels = 3
image_path = None
ckpt_path = None
history_path = None
gif_path = None
start_epoch = 0
n_epochs = 100# mi va bene globale, non devo modificarla
batch_size = 128
#sposto data loader
z_dim = 64
display_step = 400
batch_size = 2#128
# A learning rate of 0.0002 works well on DCGAN
lr = 0.0002

# These parameters control the optimizer's momentum, which you can read more about here:
# https://distill.pub/2017/momentum/
beta_1 = 0.5 
beta_2 = 0.999
device = 'cuda'
#wgan-gp
c_lambda = 10
crit_repeats = 5
##
gen = None
disc = None
gen_opt = None
disc_opt = None
criterion = None
dataloader = None
#in realtà non definirle globali, salva negli args
#<gan
def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    fakeImgs=gen(get_noise(num_images, z_dim, device=device)).detach()
    lossFake=criterion(disc(fakeImgs),torch.zeros_like(disc(fakeImgs)))
    lossReal=criterion(disc(real),torch.ones_like(disc(real)))
    disc_loss=(lossFake+lossReal)/2
    return disc_loss,fakeImgs#aggiunto ,fakeImgs
def gan_get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    fakeImgsPreds=disc(gen(get_noise(num_images, z_dim, device=device)))
    gen_loss=criterion(fakeImgsPreds,torch.ones_like(fakeImgsPreds))
    return gen_loss
#gan>
#solo per cgan<
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
#cgan>
#solo per wgan
def get_gradient(crit, real, fake, epsilon):
    mixed_images = real * epsilon + fake * (1 - epsilon)
    mixed_scores = crit(mixed_images)
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores), 
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient
def gradient_penalty(gradient):
    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty =torch.mean((gradient_norm-1)**2)
    return penalty
def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    crit_loss = torch.mean(crit_fake_pred-crit_real_pred+gp*c_lambda)
    return crit_loss
def get_gen_loss(crit_fake_pred):
    gen_loss = -torch.mean(crit_fake_pred)
    return gen_loss
##fine wgan
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
def show_tensor_images(net, image_tensor, num_images=25, size=(image_channels, image_width, image_height), nrow=5,
                           epoch_num=0, gen=None, disc=None, gen_opt=None, disc_opt=None):
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu().clamp_(0, 1)
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow, padding=0)
    permuted_image_grid = image_grid.permute(1, 2, 0).squeeze()
    image_name = f"{image_path}{epoch_num}.jpg"
    if gen!=None:
        plt.title(f"Epoch {epoch_num}")
    plt.axis("off")
    plt.imshow(permuted_image_grid)#6.4 inches by 4.8 inches
    
    #save model
    if gen!=None:
        plt.savefig(image_name)
        torch.save({
            'epoch': epoch_num,
            'gen_state_dict': gen.state_dict(),
            'disc_state_dict': disc.state_dict(),
            'gen_optimizer_state_dict': gen_opt.state_dict(),
            'disc_optimizer_state_dict': disc_opt.state_dict(),
            }, f"{ckpt_path}{net}.pkl")
        epoch_num_history = math.floor(epoch_num/5)#ogni 5 epoche checkpoint storico
        print("salvo storia per intervallo epoche ",epoch_num_history)
        torch.save({
            'epoch': epoch_num,
            'gen_state_dict': gen.state_dict(),
            'disc_state_dict': disc.state_dict(),
            'gen_optimizer_state_dict': gen_opt.state_dict(),
            'disc_opt_optimizer_state_dict': disc_opt.state_dict(),
            }, f"{history_path}{net}_ep{epoch_num_history}.pkl")
def train(net):
    cur_step = 0
    save_step = 0
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    for epoch in range(start_epoch,n_epochs):
        # Dataloader returns the batches
        for real, labels in tqdm(dataloader):
            cur_batch_size = len(real)
            #print("cur_batch_size: ",cur_batch_size)
            if net == 'gan':
                real = real.view(cur_batch_size, -1).to(device)
            else:
                real = real.to(device)
            if net == 'cgan':
                #print("shape labels",labels.shape)
                #exit()
                #RuntimeError: one_hot is only applicable to index tensor.
                one_hot_labels = get_one_hot_labels(labels.to(device), n_classes)
                image_one_hot_labels = one_hot_labels[:, :, None, None]
                image_one_hot_labels = image_one_hot_labels.repeat(1, 1, data_shape[1], data_shape[2])
            prev_fake = None#per differenza con la cgan
            prev_real = real#per differenza con la cgan
            if net == 'wgan':
                for _ in range(crit_repeats):
                    ## Update discriminator ##
                    disc_opt.zero_grad()
                    fake_noise = get_noise(cur_batch_size, z_dim, device=device)
                    fake = gen(fake_noise)
                    prev_fake = fake#per differenza con la cgan
                    disc_fake_pred = disc(fake.detach())
                    disc_real_pred = disc(real)
                    #new
                    epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
                    gradient = get_gradient(disc, real, fake.detach(), epsilon)
                    gp = gradient_penalty(gradient)
                    disc_loss = get_crit_loss(disc_fake_pred, disc_real_pred, gp, c_lambda)
                    #<old
                    #disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                    #disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
                    #disc_loss = (disc_fake_loss + disc_real_loss) / 2
                    #old>
                    mean_discriminator_loss += disc_loss.item() / display_step
                    # Update gradients
                    disc_loss.backward(retain_graph=True)
                    # Update optimizer
                    disc_opt.step()
            else:
                ## Update discriminator ##
                disc_opt.zero_grad()
                if net == 'gan':
                    disc_loss,prev_fake = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)
                    prev_fake = torch.reshape(prev_fake, (cur_batch_size, image_channels, image_width, image_height))
                    #print("gan prev_fake shape",prev_fake.shape)
                    #exit()
                else:
                    fake_noise = get_noise(cur_batch_size, z_dim, device=device)
                    if net == 'cgan':
                        fake_noise = combine_vectors(fake_noise,one_hot_labels)
                    fake = gen(fake_noise)
                    #print("fake shape",fake.shape)
                    prev_fake = fake#per differenza con la cgan
                    if net == 'cgan':
                        fake_image_and_labels = combine_vectors(fake,image_one_hot_labels)
                        #print("fake1 shape",fake.shape)
                        real_image_and_labels = combine_vectors(real,image_one_hot_labels)
                        disc_fake_pred = disc(fake_image_and_labels)
                        disc_real_pred = disc(real_image_and_labels)
                    else:
                        disc_fake_pred = disc(fake.detach())
                        disc_real_pred = disc(real)
                    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                    #print("rshape",real.shape)
                    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
                    disc_loss = (disc_fake_loss + disc_real_loss) / 2
                # Keep track of the average discriminator loss    
                mean_discriminator_loss += disc_loss.item() / display_step    
                # Update gradients
                disc_loss.backward(retain_graph=True)
                # Update optimizer
                disc_opt.step()

            ## Update generator ##
            gen_opt.zero_grad()
            if net == 'gan':
                gen_loss = gan_get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
                gen_loss.backward(retain_graph=True)
            else:
                if net == 'cgan':
                    fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)
                    disc_fake_pred = disc(fake_image_and_labels)
                    #fake_2 = combine_vectors(fake, image_one_hot_labels)#old
                    #fake_noise_2 = combine_vectors(fake_noise_2,one_hot_labels)#new
                    #fake_2 = gen(fake_noise_2)#new
                    #fake_2 = combine_vectors(fake_2, image_one_hot_labels)#new
                    #print("fake_2 shape",fake.shape)
                    #exit()
                else:
                    fake_noise_2 = get_noise(cur_batch_size, z_dim, device=device)#new per cgan
                    fake_2 = gen(fake_noise_2)
                    disc_fake_pred = disc(fake_2)
                if net == 'wgan':
                    gen_loss = get_gen_loss(disc_fake_pred)
                else:
                    gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
                gen_loss.backward()
            gen_opt.step()
            
            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item() / display_step
            #print("cur_step",cur_step,"display_step",display_step)
            ## Visualization code ##
            if cur_step % display_step == 0 and cur_step > 0:
                print(f"Epoch {epoch} Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
                #show_tensor_images(fake,name=epoch,gen=gen,disc=disc)
                print("plot ",prev_fake.shape)
                show_tensor_images(net, prev_fake, epoch_num=epoch, gen=gen, disc=disc, gen_opt=gen_opt, disc_opt=disc_opt)
                save_step += 1
                show_tensor_images(net, prev_real)
                mean_generator_loss = 0
                mean_discriminator_loss = 0
            cur_step += 1   
    print("Addestramento concluso.")

def generate_gif():
    #alla fine dell'addestramento genero una gif che mostra visivamente i progressi della rete
    filenames = [f for f in listdir(image_path) if isfile(join(image_path, f))]
    filenames = sorted(filenames,key=lambda x: int(x.split('.')[0]))
    with imageio.get_writer(f'{gif_path}movie_{n_epochs}.gif', mode='I', duration = 0.5) as writer:
        for filename in filenames:
            file_path = f"{image_path}{filename}"
            #print("file_path",file_path)
            image = imageio.imread(file_path)
            #print(image)
            writer.append_data(image)

def preprocess(img):
    #img.unsqueeze(0)
    img = torch.nn.functional.interpolate(img, size=(299, 299), mode='bilinear', align_corners=False)
    return img

def get_inception_model():
    inception_path = "inception_v3_google-1a9a5a14.pth"
    inception_model = inception_v3(pretrained=False)
    inception_model.load_state_dict(torch.load(inception_path))
    inception_model.to(device)
    inception_model = inception_model.eval()
    inception_model.fc = torch.nn.Identity()
    return inception_model

def matrix_sqrt(x):
    y = x.cpu().detach().numpy()
    y = scipy.linalg.sqrtm(y)
    return torch.Tensor(y.real, device=x.device)
def frechet_distance(mu_x, mu_y, sigma_x, sigma_y):
    return (mu_x - mu_y).dot(mu_x - mu_y) + torch.trace(sigma_x) + torch.trace(sigma_y) - 2 * torch.trace(matrix_sqrt(sigma_x @ sigma_y))
def get_covariance(features):
    return torch.Tensor(np.cov(features.detach().numpy(), rowvar=False))    

def get_fid(net):
    #image_size = 299
    device = 'cuda'

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = CelebACustom(download=False,transform=transform)
    dataset.__len__()

    fake_features_list = []
    real_features_list = []

    gen.eval()
    n_samples = 512 # The total number of samples
    batch_size = 4 # Samples per iteration

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True)

    cur_samples = 0
    inception_model = get_inception_model()
    with torch.no_grad(): # You don't need to calculate gradients here, so you do this to save memory
        try:
            for real_example, _ in tqdm(dataloader, total=n_samples // batch_size): # Go by batch
                #!nvidia-smi
                cur_batch_size = len(real_example)
                real_samples = real_example
                real_features = inception_model(real_samples.to(device)).detach().to('cpu') # Move features to CPU
                real_features_list.append(real_features)
                #print("len real_example",len(real_example))
                #print("z_dim",z_dim)
                
                fake_samples = get_noise(len(real_example), z_dim).to(device)
                #print("shape fake_samples",fake_samples.shape)
                fake_samples = gen(fake_samples)
                #print("generated",fake_samples.shape)
                if net == 'gan':
                    fake_samples = torch.reshape(fake_samples, (batch_size, image_channels, image_width, image_height))
                fake_samples = preprocess(fake_samples)
                #print("shape fake_samples",fake_samples.shape)
                fake_features = inception_model(fake_samples.to(device)).detach().to('cpu')
                #print("shape fake_features",fake_features.shape)
                fake_features_list.append(fake_features)
                cur_samples += len(real_samples)
                #print(f"{cur_samples}//{n_samples}")
                if cur_samples >= n_samples:
                    break
        except Exception as e:
            print(e)
            print("Error in loop")

    #print(fake_features_list)
    fake_features_all = torch.cat(fake_features_list)
    real_features_all = torch.cat(real_features_list)

    mu_fake = fake_features_all.mean(0)
    mu_real = real_features_all.mean(0)
    sigma_fake = get_covariance(fake_features_all)
    sigma_real = get_covariance(real_features_all)

    indices = [2, 4, 5]
    fake_dist = MultivariateNormal(mu_fake[indices], sigma_fake[indices][:, indices])
    fake_samples = fake_dist.sample((5000,))
    real_dist = MultivariateNormal(mu_real[indices], sigma_real[indices][:, indices])
    real_samples = real_dist.sample((5000,))

    df_fake = pd.DataFrame(fake_samples.numpy(), columns=indices)
    df_real = pd.DataFrame(real_samples.numpy(), columns=indices)
    df_fake["is_real"] = "no"
    df_real["is_real"] = "yes"
    df = pd.concat([df_fake, df_real])
    sns.pairplot(df, plot_kws={'alpha': 0.1}, hue='is_real')

    with torch.no_grad():
        print("La FID per 5000 dati campione è: ",frechet_distance(mu_real, mu_fake, sigma_real, sigma_fake).item())


@click.command()
@click.pass_context
# General options.
@click.option('--outdir', help='Dove salvare i risultati', required=True, metavar='DIR')
@click.option('--net', help='La tipologia di GAN da addestrare', type=click.Choice(['gan','dcgan', 'wgan', 'cgan','cyclegan', 'progan', 'sga2']))
def main(ctx, outdir, net):
    global gen, disc, gen_opt, disc_opt, criterion
    global image_path, ckpt_path, history_path, gif_path
    global image_width, image_height, n_classes, data_shape#per cifar-10 3x32x32
    global dataloader
    global lr

    data_shape = (3, 64, 64)#3,32,32
    dataset = None
    t_image_width = 64
    batch_size = 25
    print(f"Caricamento dati...")
    #if net == 'cgan':
    #    t_image_width = 32
    transform = transforms.Compose([
        transforms.Resize(t_image_width),
        transforms.CenterCrop(t_image_width),
        #transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    if net == 'cgan':
        #dataset = MNIST('.', download=False, transform=transform)
        from torchvision.datasets import CIFAR10
        dataset = CIFAR10('.', download=False, transform=transform)
    else:
        #print("isdir",os.path.isdir("Downloads/datasets/reduced_celeba/reduced_celeba"))
        dataset = CelebACustom(download=False,transform=transform)
    print(dataset[0][0].shape)
    dataset.__len__()
    dataset.__getitem__(1)[0].shape
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True)
    ##
    #return
    if net == None:
        net = 'dcgan'
    outdir = f"results/{outdir}/{net}"
    print(f"Avvio addestramento di una {net}. I risultati verrano salvati nella cartella {outdir}")
    #basedir = f'{outdir}_{net}'#"Downloads/gan_results/dcgan/"
    image_path = f'{outdir}/img/'
    ckpt_path = f'{outdir}/checkpoints/'
    history_path = f'{outdir}/history/'
    gif_path = f'{outdir}/gif/'
    list_paths = [image_path,ckpt_path,history_path,gif_path]#outdir
    for path_el in list_paths:
        if not os.path.exists(path_el):
            os.makedirs(path_el)
    insize_g = (64,)#unused
    insize_d = (3,64,64)#unused
    modules = [f"models.{net}.networks.Generator",f"models.{net}.networks.Discriminator"]
    for module in modules:
        import_module(module)
    Generator = getattr(import_module(f"models.{net}.networks.Generator"), 'Generator')
    Discriminator = getattr(import_module(f"models.{net}.networks.Discriminator"), 'Discriminator')
    criterion = nn.BCEWithLogitsLoss()
    gen_input_dim = z_dim
    disc_input_dim = image_channels
    if net == 'gan':
        disc_input_dim = image_channels*image_height*image_width
    if net == 'cgan':
        gen_input_dim, disc_input_dim = get_input_dimensions(z_dim, data_shape, n_classes)
    print("disc_input_dim",disc_input_dim)
    gen = Generator(input_dim=gen_input_dim).to(device)
    disc = Discriminator(disc_input_dim).to(device) 
    if net == 'gan':
        lr = 0.00001
        gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
        disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
        insize_g = (64,)
        insize_d = (3*64*64,)
        print("Architettura Generatore")
        #summary(gen, insize_g)
        summary(gen, (64,))
        print("Architettura Discriminatore")
        #summary(disc, insize_d)
        summary(disc, (3*64*64,))
    if net == 'dcgan':
        gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
        disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))
        print("Architettura Generatore")
        #summary(gen, insize_g)
        summary(gen, (64,))
        print("Architettura Discriminatore")
        #summary(disc, insize_d)
        summary(disc, (3,64,64))
    if net == 'wgan':
        gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
        disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))
        print("Architettura Generatore")
        #summary(gen, insize_g)
        summary(gen, (64,))
        print("Architettura Discriminatore")
        #summary(disc, insize_d)
        summary(disc, (3,64,64)) 
    if net == 'cgan':
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
    #exit("fine...")
    preload_model = len(os.listdir(history_path)) >= 1#ckpt_path
    print("preload_model",preload_model,"history_path",os.listdir(history_path))
    #return
    # You initialize the weights to the normal distribution
    # with mean 0 and standard deviation 0.02

    if preload_model:
        ##loading state...
        checkpoint = torch.load(f"{ckpt_path}{net}.pkl")
        gen.load_state_dict(checkpoint['gen_state_dict'])
        gen_opt.load_state_dict(checkpoint['gen_optimizer_state_dict'])
        gen.eval()
        disc.load_state_dict(checkpoint['disc_state_dict'])
        disc_opt.load_state_dict(checkpoint['disc_optimizer_state_dict'])
        disc.eval()
        start_epoch = checkpoint['epoch']
        print("start epoch",start_epoch,"type",type(start_epoch))
    else:
        gen = gen.apply(weights_init)
        disc = disc.apply(weights_init)

    #summary
    #print(gen)

    cur_batch_size = 25
    """TEST
    #z_dim => gen_input_dim
    fake_noise = get_noise(cur_batch_size, gen_input_dim, device=device)
    fake = gen(fake_noise)
    #print("shape1",fake.shape)
    show_tensor_images(net, fake)

    #image_size = 299
    if net == 'gan':#non ho convoluzioni, ma solo linear layer (1D: C*W*H)
        fake = torch.reshape(fake, (cur_batch_size, image_channels, image_width, image_height))
    #fake = fake.view(25, 3, 64, 64)
    #print("shape2",fake.shape)
    fake = preprocess(fake)
    #print("shape3",fake.shape)
    show_tensor_images(net, fake)
    """
    ##
    train(net)
    #exit()
    print("Generazione gif...")
    generate_gif()
    print("Ottengo la fid...")
    print("Questa operazione potrebbe richiedere qualche minuto")  
    get_fid(net)

    
if __name__ == "__main__":
    main()