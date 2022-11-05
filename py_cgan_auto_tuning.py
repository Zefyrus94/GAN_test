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
print("cuda available?",torch.cuda.is_available())
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
n_epochs = 12#200
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
def create_data_loader_mnist():
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
def get_fid(gen):
    #image_size = 299
    device = 'cuda'
    dataloader = create_data_loader_mnist()
    fake_features_list = []
    real_features_list = []
    gen_copy = copy.deepcopy(gen)
    gen_copy.eval()#for inference
    n_samples = 512 # The total number of samples
    batch_size = 4 # Samples per iteration
    cur_samples = 0
    print("Recupero il modello InceptionV3...")
    inception_model = get_inception_model()
    print("Valutazione in corso...")
    with torch.no_grad(): # You don't need to calculate gradients here, so you do this to save memory
        try:
            for real_example, labels in tqdm(dataloader, total=n_samples // batch_size): # Go by batch
                #!nvidia-smi
                print("real...")
                cur_batch_size = len(real_example)
                real_samples = real_example
                real_features = inception_model(real_samples.to(device)).detach().to('cpu') # Move features to CPU
                real_features_list.append(real_features)
                #print("len real_example",len(real_example))
                #print("z_dim",z_dim)
                print("fake...")
                fake_samples = get_noise(cur_batch_size, z_dim).to(device)
                one_hot_labels = get_one_hot_labels(labels.to(device), n_classes)
                fake_samples = combine_vectors(fake_samples,one_hot_labels)
                #print("shape fake_samples",fake_samples.shape)
                fake_samples = gen(fake_samples)
                print("generated",fake_samples.shape)
                print("preprocess fake....")
                fake_samples = preprocess(fake_samples)
                print("shape fake_samples",fake_samples.shape)
                fake_features = inception_model(fake_samples.to(device)).detach().to('cpu')
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
        print("La FID per 5000 dati campione è: ",frechet_distance(mu_real, mu_fake, sigma_real, sigma_fake).item())
###FINE FID
def train(config):
    #https://discuss.ray.io/t/runtimeerror-no-cuda-gpus-are-available/1787
    assert torch.cuda.is_available()
    dataloader = create_data_loader_mnist()
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
    loss_f = open("loss.txt", "a")
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
            fid = get_fid(gen)#randrange(10)
            tune.report(fid=fid, loss_d=(running_loss_d / num_of_batches), loss_g=(running_loss_g / num_of_batches))
        print(f'[Epoch {epoch + 1}/{n_epochs}] loss d: {running_loss_d / num_of_batches:.3f}; loss g: {running_loss_g / num_of_batches:.3f}')
        loss_f.write(f"{running_loss_d / num_of_batches:.3f};{running_loss_g / num_of_batches:.3f}\n")
        loss_f.close()
#main
if __name__ == '__main__':
    start = time.time()
    PATH_D = './mnist_disc.pth'
    PATH_G = './mnist_gen.pth'
    #search space
    #,"batch_size": tune.choice([2, 4, 8, 16])
    config = {
        "hidden_dim": 64,#tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
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
        max_failures=10,#0, # set this to a large value, 100 works in my case
        resources_per_trial={"cpu": 2, "gpu": 1},#o non vede la gpu (cuda.device)
        config=config,
        num_samples=10,#num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("fid", "min", "last")#loss_g
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final fid: {}".format(best_trial.last_result["fid"]))
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