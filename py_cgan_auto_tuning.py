#Total elapsed time: 3833.64 seconds,     Train 200 epochs 3831.86 seconds
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
#https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
#prove
from random import randrange
#fid
from torchvision.models import inception_v3
from torch.distributions import MultivariateNormal
import zipfile
import scipy
import numpy as np
#clone modello
import copy
torch.manual_seed(0) # Set for our testing purposes, please do not change!
#print("cuda available?",torch.cuda.is_available())
####per la valutazione: VAE...
#https://colab.research.google.com/github/smartgeometry-ucl/dl4g/blob/master/variational_autoencoder.ipynb#scrollTo=yOzgLfh3bwtR
"""
Encoder(
  (conv1): Conv2d(1, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
  (conv2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
  (fc_mu): Identity()
  (fc_logvar): Identity()
)
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 14, 14]           1,088
            Conv2d-2            [-1, 128, 7, 7]         131,200
          Identity-3                 [-1, 6272]               0
          Identity-4                 [-1, 6272]               0
================================================================
Total params: 132,288
Trainable params: 132,288
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.24
Params size (MB): 0.50
Estimated Total Size (MB): 0.75
----------------------------------------------------------------
"""
latent_dims = 2
#num_epochs = 100
#batch_size = 128
capacity = 64
#learning_rate = 1e-3
#variational_beta = 1
#use_gpu = True
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        c = capacity
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=4, stride=2, padding=1) # out: c x 14 x 14
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=4, stride=2, padding=1) # out: c x 7 x 7
        self.fc_mu = nn.Linear(in_features=c*2*7*7, out_features=latent_dims)
        self.fc_logvar = nn.Linear(in_features=c*2*7*7, out_features=latent_dims)
            
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        c = capacity
        self.fc = nn.Linear(in_features=latent_dims, out_features=c*2*7*7)
        self.conv2 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)
            
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), capacity*2, 7, 7) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv1(x)) # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        return x
    
class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar
    
    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

####fine vae
class Generator(nn.Module):
    '''
    Generator Class
    Values:
        input_dim: the dimension of the input vector, a scalar
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, input_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(input_dim, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )
    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a generator block of DCGAN;
        a transposed convolution, a batchnorm (except in the final layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh(),
            )
    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, input_dim)
        '''
        x = noise.view(len(noise), self.input_dim, 1, 1)
        return self.gen(x)
def get_noise(n_samples, input_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, input_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        input_dim: the dimension of the input vector, a scalar
        device: the device type
    '''
    return torch.randn(n_samples, input_dim, device=device)
class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
      im_chan: the number of channels in the images, fitted for the dataset used, a scalar
            (MNIST is black-and-white, so 1 channel is your default)
      hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, im_chan=1, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.make_disc_block(im_chan, hidden_dim),
            self.make_disc_block(hidden_dim, hidden_dim * 2),
            self.make_disc_block(hidden_dim * 2, 1, final_layer=True),
        )
    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a discriminator block of the DCGAN; 
        a convolution, a batchnorm (except in the final layer), and an activation (except in the final layer).
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            )
    def forward(self, image):
        '''
        Function for completing a forward pass of the discriminator: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_chan)
        '''
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)
# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_one_hot_labels
import torch.nn.functional as F
def get_one_hot_labels(labels, n_classes):
    '''
    Function for creating one-hot vectors for the labels, returns a tensor of shape (?, num_classes).
    Parameters:
        labels: tensor of labels from the dataloader, size (?)
        n_classes: the total number of classes in the dataset, an integer scalar
    '''
    #### START CODE HERE ####
    return F.one_hot(labels, n_classes)
    #### END CODE HERE ####
# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: combine_vectors
def combine_vectors(x, y):
    '''
    Function for combining two vectors with shapes (n_samples, ?) and (n_samples, ?).
    Parameters:
      x: (n_samples, ?) the first vector. 
        In this assignment, this will be the noise vector of shape (n_samples, z_dim), 
        but you shouldn't need to know the second dimension's size.
      y: (n_samples, ?) the second vector.
        Once again, in this assignment this will be the one-hot class vector 
        with the shape (n_samples, n_classes), but you shouldn't assume this in your code.
    '''
    # Note: Make sure this function outputs a float no matter what inputs it receives
    #### START CODE HERE ####
    combined = torch.cat((x.float(), y.float()), 1)
    #### END CODE HERE ####
    return combined
mnist_shape = (1, 28, 28)
n_classes = 10
criterion = nn.BCEWithLogitsLoss()
n_epochs = 100#200
z_dim = 64
display_step = 500
batch_size = 128
lr = 0.0002
device = 'cuda'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
dataloader = DataLoader(
    MNIST('.', download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True)
# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_input_dimensions
def get_input_dimensions(z_dim, mnist_shape, n_classes):
    '''
    Function for getting the size of the conditional input dimensions 
    from z_dim, the image shape, and number of classes.
    Parameters:
        z_dim: the dimension of the noise vector, a scalar
        mnist_shape: the shape of each MNIST image as (C, W, H), which is (1, 28, 28)
        n_classes: the total number of classes in the dataset, an integer scalar
                (10 for MNIST)
    Returns: 
        generator_input_dim: the input dimensionality of the conditional generator, 
                          which takes the noise and class vectors
        discriminator_im_chan: the number of input channels to the discriminator
                            (e.g. C x 28 x 28 for MNIST)
    '''
    #### START CODE HERE ####
    generator_input_dim = z_dim + n_classes
    discriminator_im_chan = mnist_shape[0] + n_classes
    #### END CODE HERE ####
    return generator_input_dim, discriminator_im_chan
def create_data_loader_mnist(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    #batch_size = 256
    mnist_data = MNIST(root='./data', download=True, transform=transform)
    #num_workers=16, pin_memory=True
    dataloader = DataLoader(mnist_data, batch_size=batch_size,shuffle=True,num_workers=16)
    return dataloader
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
###FID
def preprocess(img):
    #img.unsqueeze(0)
    img = torch.nn.functional.interpolate(img, size=(299, 299), mode='bilinear', align_corners=False)
    return img
def get_vae_encoder():
    #inception_zip_path = "inception_v3_google-1a9a5a14.zip"
    #with zipfile.ZipFile(inception_zip_path, 'r') as zip_ref:
    #    zip_ref.extractall('.')
    vae_path = "/home/nvidia/workspace/giacomobonanni/GAN_test/vae_model.pth"
    state_dict = torch.load(vae_path)
    vae_fid = VariationalAutoencoder()
    vae_fid.load_state_dict(torch.load(vae_path))
    vae_fid.to(device)
    encoder = vae_fid.encoder
    encoder = encoder.eval()
    encoder.fc_mu = torch.nn.Identity()
    #encoder.fc_logvar = torch.nn.Identity()
    return encoder
def get_inception_model():
    #inception_v3_google-1a9a5a14.pth
    #TUNE_ORIG_WORKING_DIR
    #https://discuss.ray.io/t/filenotfounderror-after-i-used-tune-run/5747
    #ABSOLUTE PATH
    #inception_zip_path = "inception_v3_google-1a9a5a14.zip"
    inception_zip_path = "/home/nvidia/workspace/giacomobonanni/GAN_test/inception_v3_google-1a9a5a14.zip"
    with zipfile.ZipFile(inception_zip_path, 'r') as zip_ref:
        zip_ref.extractall('.')
    inception_path = "inception_v3_google-1a9a5a14.pth"
    #exit("prova unzip")
    inception_model = inception_v3(pretrained=False)
    inception_model.load_state_dict(torch.load(inception_path))
    inception_model.to(device)
    inception_model = inception_model.eval()#for inference
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
"""
NOTE SULLA FID
https://wandb.ai/ayush-thakur/gan-evaluation/reports/How-to-Evaluate-GANs-using-Frechet-Inception-Distance-FID---Vmlldzo0MTAxOTI
Observations
    The FID score decreases with the better model checkpoint. One can pick a model checkpoint that generates a low FID score for inference.
    The FID score is relatively high because the Inception model is trained on Imagenet which constitutes natural images while our GAN is trained on the FashionMNIST dataset.
Shortcomings of FID
    It uses a pre-trained Inception model, which may not capture all features. This can result in a high FID score as in the above case.
    It needs a large sample size. The minimum recommended sample size is 10,000. For a high-resolution image(say 512x512 pixels) this can be computationally expensive and slow to run.
    Limited statistics(mean and covariance) are used to compute the FID score.
"""
def get_fid(gen,batch_size):
    #image_size = 299
    device = 'cuda'
    dataloader = create_data_loader_mnist(batch_size)
    fake_features_list = []
    real_features_list = []
    gen_copy = copy.deepcopy(gen)
    gen_copy.eval()#for inference
    n_samples = 512 # The total number of samples
    batch_size = 4 # Samples per iteration
    cur_samples = 0
    #print("Recupero l'encoder del VAE...")
    encoder_fid = get_vae_encoder()
    #print("Valutazione in corso...")
    with torch.no_grad(): # You don't need to calculate gradients here, so you do this to save memory
        try:
            for real_example, labels in tqdm(dataloader, total=n_samples // batch_size): # Go by batch
                #!nvidia-smi
                #print("real...")
                cur_batch_size = len(real_example)
                real_samples = real_example
                #print("chiamo inception model...")
                #[0] perché restituisce mu, log_var
                real_features = encoder_fid(real_samples.to(device))[0].detach().to('cpu') # Move features to CPU
                #print("chiamata conclusa a vae model...")
                #print("real_features",real_features)
                real_features_list.append(real_features)
                #print("len real_example",len(real_example))
                #print("z_dim",z_dim)
                #print("fake...")
                fake_samples = get_noise(cur_batch_size, z_dim).to(device)
                one_hot_labels = get_one_hot_labels(labels.to(device), n_classes)
                fake_samples = combine_vectors(fake_samples,one_hot_labels)
                #print("shape fake_samples",fake_samples.shape)
                fake_samples = gen(fake_samples)
                #print("generated",fake_samples.shape)
                #print("preprocess fake....")
                #niente preprocessing fake(1,28,28), real (1,28,28), input vae (1,28,28), tutto ok
                #fake_samples = preprocess(fake_samples)
                #print("shape fake_samples",fake_samples.shape)
                fake_features = encoder_fid(fake_samples.to(device))[0].detach().to('cpu')
                #print("shape fake_features",fake_features.shape)
                fake_features_list.append(fake_features)
                cur_samples += len(real_samples)
                #print(f"{cur_samples}//{n_samples}")
                if cur_samples >= n_samples:
                    break
        except Exception as e:
            print(e)
            exit("Error in loop")
    #print(fake_features_list)
    fake_features_all = torch.cat(fake_features_list)
    real_features_all = torch.cat(real_features_list)
    mu_fake = fake_features_all.mean(0)
    mu_real = real_features_all.mean(0)
    sigma_fake = get_covariance(fake_features_all)
    sigma_real = get_covariance(real_features_all)
    with torch.no_grad():
        fid = frechet_distance(mu_real, mu_fake, sigma_real, sigma_fake).item()
        #print("La FID è: ",fid)
        return fid
###FINE FID
def train(config):
    #https://discuss.ray.io/t/runtimeerror-no-cuda-gpus-are-available/1787
    assert torch.cuda.is_available()
    batch_size = config["batch_size"]#.sample()#spostato
    dataloader = create_data_loader_mnist(batch_size)
    generator_input_dim, discriminator_im_chan = get_input_dimensions(z_dim, mnist_shape, n_classes)
    gen = Generator(input_dim=generator_input_dim, hidden_dim=config["hidden_dim"]).to(device)
    #https://discuss.pytorch.org/t/syntax-error-on-ray-ray-tune-not-supported-between-instances-of-float-and-float/144693
    gen_opt = torch.optim.Adam(gen.parameters(), lr=config["lr_g"])#=lr#.sample()
    disc = Discriminator(im_chan=discriminator_im_chan, hidden_dim=config["hidden_dim"]).to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=config["lr_d"])#=lr#.sample()
    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)
    checkpoint_g_path = None
    if checkpoint_g_path:
        gen_state, gen_opt_state = torch.load(checkpoint_g_path)
        gen.load_state_dict(gen_state)
        gen_opt.load_state_dict(gen_opt_state)
        disc_state, disc_opt_state = torch.load(checkpoint_d_path)
        disc.load_state_dict(disc_state)
        disc_opt.load_state_dict(disc_opt_state)
    ##
    #loss_f = open("loss.txt", "a")
    # UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
    # GRADED CELL
    cur_step = 0
    generator_losses = []
    discriminator_losses = []
    #UNIT TEST NOTE: Initializations needed for grading
    noise_and_labels = False
    fake = False
    fake_image_and_labels = False
    real_image_and_labels = False
    disc_fake_pred = False
    disc_real_pred = False
    num_of_batches = len(dataloader)
    for epoch in range(n_epochs):
        running_loss_d = 0.0
        running_loss_g = 0.0
        # Dataloader returns the batches and the labels
        for real, labels in tqdm(dataloader):
            cur_batch_size = len(real)
            # Flatten the batch of real images from the dataset
            real = real.to(device)
            one_hot_labels = get_one_hot_labels(labels.to(device), n_classes)
            image_one_hot_labels = one_hot_labels[:, :, None, None]
            image_one_hot_labels = image_one_hot_labels.repeat(1, 1, mnist_shape[1], mnist_shape[2])
            ### Update discriminator ###
            # Zero out the discriminator gradients
            disc_opt.zero_grad()
            # Get noise corresponding to the current batch_size 
            fake_noise = get_noise(cur_batch_size, z_dim, device=device)

            # Now you can get the images from the generator
            # Steps: 1) Combine the noise vectors and the one-hot labels for the generator
            #        2) Generate the conditioned fake images
            #### START CODE HERE ####
            noise_and_labels = combine_vectors(fake_noise, one_hot_labels)
            fake = gen(noise_and_labels)
            #### END CODE HERE ####
            # Make sure that enough images were generated
            assert len(fake) == len(real)
            # Check that correct tensors were combined
            assert tuple(noise_and_labels.shape) == (cur_batch_size, fake_noise.shape[1] + one_hot_labels.shape[1])
            # It comes from the correct generator
            assert tuple(fake.shape) == (len(real), 1, 28, 28)
            # Now you can get the predictions from the discriminator
            # Steps: 1) Create the input for the discriminator
            #           a) Combine the fake images with image_one_hot_labels, 
            #              remember to detach the generator (.detach()) so you do not backpropagate through it
            #           b) Combine the real images with image_one_hot_labels
            #        2) Get the discriminator's prediction on the fakes as disc_fake_pred
            #        3) Get the discriminator's prediction on the reals as disc_real_pred
            #### START CODE HERE ####
            fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)
            real_image_and_labels = combine_vectors(real, image_one_hot_labels)
            disc_fake_pred = disc(fake_image_and_labels.detach())
            disc_real_pred = disc(real_image_and_labels)
            #### END CODE HERE ####
            # Make sure shapes are correct 
            assert tuple(fake_image_and_labels.shape) == (len(real), fake.detach().shape[1] + image_one_hot_labels.shape[1], 28 ,28)
            assert tuple(real_image_and_labels.shape) == (len(real), real.shape[1] + image_one_hot_labels.shape[1], 28 ,28)
            # Make sure that enough predictions were made
            assert len(disc_real_pred) == len(real)
            # Make sure that the inputs are different
            assert torch.any(fake_image_and_labels != real_image_and_labels)
            # Shapes must match
            assert tuple(fake_image_and_labels.shape) == tuple(real_image_and_labels.shape)
            assert tuple(disc_fake_pred.shape) == tuple(disc_real_pred.shape)
            disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_loss.backward(retain_graph=True)
            disc_opt.step() 
            # Keep track of the average discriminator loss
            discriminator_losses += [disc_loss.item()]
            running_loss_d += disc_loss.item()
            ### Update generator ###
            # Zero out the generator gradients
            gen_opt.zero_grad()
            fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)
            # This will error if you didn't concatenate your labels to your image correctly
            disc_fake_pred = disc(fake_image_and_labels)
            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
            gen_loss.backward()
            gen_opt.step()
            # Keep track of the generator losses
            generator_losses += [gen_loss.item()]
            running_loss_g += gen_loss.item()
            if cur_step % display_step == 0 and cur_step > 0:
                gen_mean = sum(generator_losses[-display_step:]) / display_step
                disc_mean = sum(discriminator_losses[-display_step:]) / display_step
                print(f"Step {cur_step}: Generator loss: {gen_mean}, discriminator loss: {disc_mean}")
                """
                show_tensor_images(fake)
                show_tensor_images(real)
                step_bins = 20
                x_axis = sorted([i * step_bins for i in range(len(generator_losses) // step_bins)] * step_bins)
                num_examples = (len(generator_losses) // step_bins) * step_bins
                plt.plot(
                    range(num_examples // step_bins), 
                    torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="Generator Loss"
                )
                plt.plot(
                    range(num_examples // step_bins), 
                    torch.Tensor(discriminator_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="Discriminator Loss"
                )
                plt.legend()
                plt.show()
                """
            elif cur_step == 0:
                print("Congratulations! If you've gotten here, it's working. Please let this train until you're happy with how the generated numbers look, and then go on to the exploration!")
            cur_step += 1
            #with tune.checkpoint_dir(epoch) as checkpoint_dir:
            #    path = os.path.join(checkpoint_dir, "checkpoint")
            #    torch.save((net.state_dict(), optimizer.state_dict()), path)
            #prove fid
            """
            #sposto il reporting fuori, così lo faccio per n_epochs
            fid = get_fid(gen,batch_size)#randrange(10)#
            loss_d = (running_loss_d / num_of_batches)
            loss_g = (running_loss_g / num_of_batches)
            tune.report(fid=fid, loss_d=loss_d, loss_g=loss_g)
            """
        print(f'[Epoch {epoch + 1}/{n_epochs}] loss d: {running_loss_d / num_of_batches:.3f}; loss g: {running_loss_g / num_of_batches:.3f}')
        loss_f = open("loss.txt", "a")
        loss_f.write(f"{running_loss_d / num_of_batches:.3f};{running_loss_g / num_of_batches:.3f}\n")
        loss_f.close()
    fid = get_fid(gen,batch_size)#randrange(10)#
    loss_d = (running_loss_d / num_of_batches)
    loss_g = (running_loss_g / num_of_batches)
    tune.report(fid=fid, loss_d=loss_d, loss_g=loss_g)
#main
"""
2022-11-06 16:06:53,770 ERROR trial_runner.py:987 -- Trial train_9f698_00001: Error processing event.
ray.exceptions.RayTaskError(ValueError): ray::ImplicitFunc.train() (pid=2270450, ip=193.204.161.50, repr=train)
  File "/home/nvidia/anaconda3/envs/pytorch1.13/lib/python3.9/site-packages/ray/tune/trainable/trainable.py", line 349, in train
    result = self.step()
  File "/home/nvidia/anaconda3/envs/pytorch1.13/lib/python3.9/site-packages/ray/tune/trainable/function_trainable.py", line 417, in step
    self._report_thread_runner_error(block=True)
  File "/home/nvidia/anaconda3/envs/pytorch1.13/lib/python3.9/site-packages/ray/tune/trainable/function_trainable.py", line 589, in _report_thread_runner_error
    raise e
  File "/home/nvidia/anaconda3/envs/pytorch1.13/lib/python3.9/site-packages/ray/tune/trainable/function_trainable.py", line 289, in run
    self._entrypoint()
  File "/home/nvidia/anaconda3/envs/pytorch1.13/lib/python3.9/site-packages/ray/tune/trainable/function_trainable.py", line 362, in entrypoint
    return self._trainable_func(
  File "/home/nvidia/anaconda3/envs/pytorch1.13/lib/python3.9/site-packages/ray/tune/trainable/function_trainable.py", line 684, in _trainable_func
    output = fn()
  File "/home/nvidia/workspace/giacomobonanni/GAN_test/py_cgan_auto_tuning.py", line 428, in train
    dataloader = create_data_loader_mnist()
  File "/home/nvidia/workspace/giacomobonanni/GAN_test/py_cgan_auto_tuning.py", line 300, in create_data_loader_mnist
    dataloader = DataLoader(mnist_data, batch_size=config["batch_size"],shuffle=True,num_workers=16)
  File "/home/nvidia/anaconda3/envs/pytorch1.13/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 350, in __init__
    batch_sampler = BatchSampler(sampler, batch_size, drop_last)
  File "/home/nvidia/anaconda3/envs/pytorch1.13/lib/python3.9/site-packages/torch/utils/data/sampler.py", line 232, in __init__
    raise ValueError("batch_size should be a positive integer value, "
ValueError: batch_size should be a positive integer value, but got batch_size=<ray.tune.search.sample.Categorical object at 0x7f809c36fc10>
"""
#https://www.analyticsvidhya.com/blog/2021/05/tuning-the-hyperparameters-and-layers-of-neural-network-deep-learning/
if __name__ == '__main__':
    start = time.time()
    PATH_D = './mnist_disc.pth'
    PATH_G = './mnist_gen.pth'
    #search space
    #,"batch_size": tune.choice([2, 4, 8, 16])
    config = {
        "hidden_dim": tune.sample_from(lambda _: 2**np.random.randint(6, 9)),#64,#64=>512
        "batch_size": tune.sample_from(lambda _: 2**np.random.randint(5, 8)),#tune.choice([32,64,128,256]),
        "lr_g": tune.loguniform(1e-4, 1e-1),
        "lr_d": tune.loguniform(1e-4, 1e-1)
    }
    ##
    scheduler = ASHAScheduler(
        metric="loss_g",
        mode="min",
        max_t=n_epochs,#max_num_epochs
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        #metric_columns=["loss_d", "loss_g"]
        )
    #https://docs.ray.io/en/latest/tune/api_docs/reporters.html
    # Add a custom metric column, in addition to the default metrics.
    # Note that this must be a metric that is returned in your training results.
    #reporter.add_metric_column("loss_d")
    result = tune.run(
        #partial(train_cifar, data_dir=data_dir),
        train,
        max_failures=1,#0, # set this to a large value, 100 works in my case
        resources_per_trial={"cpu": 2, "gpu": 1},#o non vede la gpu (cuda.device)
        config=config,
        num_samples=16,#3#num_samples,#il numero di permutazioni/tentativi che farò
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("fid", "min", "last")#loss_g
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final fid: {}".format(best_trial.last_result["fid"]))
    randnum = randrange(1000)
    best_trial_f = open(f"best_trial_{randnum}.txt", "a")
    best_trial_f.write("Best trial config: {}".format(best_trial.config))
    best_trial_f.write("Best trial final fid: {}".format(best_trial.last_result["fid"]))
    best_trial_f.close()
    #print("Best trial final validation accuracy: {}".format(
    #    best_trial.last_result["accuracy"]))
    #best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    #train
    """
    start_train = time.time()
    loss_f = open("loss.txt", "a")
    train(gen, disc, dataloader, loss_f)
    loss_f.close()
    end_train = time.time()
    # save
    torch.save(gen.state_dict(), PATH_G)
    torch.save(disc.state_dict(), PATH_D)
    end = time.time()
    seconds = (end - start)
    seconds_train = (end_train - start_train)
    print(f"Total elapsed time: {seconds:.2f} seconds, \
     Train {n_epochs} epochs {seconds_train:.2f} seconds") 
    """
    print("finished.")