import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
#from modules import UNet
from modules_split import UNet
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda:1"):#cuda => cuda:1
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                device = x.device
                alpha = alpha.to(device)
                alpha_hat = alpha_hat.to(device)
                predicted_noise = predicted_noise.to(device)
                beta = beta.to(device)
                noise = noise.to(device)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet(device=device)#to(device) per i pesi della rete
    #####
    #"""
    from torchsummary import summary
    device_sum = "cuda"
    model_sum = UNet(device=device_sum)#.to(device_sum)
    summary(model_sum, [(3, 64, 64),()])
    print("end summary")
    return
    #"""
    #####
    start_epoch = 1#agg
    #start_epoch = 278#agg
    """
    ckpt = torch.load("./models/DDPM_Uncondtional/ckpt.pt")#agg
    model.load_state_dict(ckpt)#.state_dict()
    """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # (manca lo state dell'optimizer)
    #model.train()#agg

    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    for epoch in range(start_epoch,args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            #x_t = x_t.to(device)#n
            #noise = noise.to(device)#n
            #print("x_t dev: ",x_t.device.index)
            #print("t dev: ",t.device.index)
            #print("model input",x_t.device,t.device)
            predicted_noise = model(x_t, t)
            #RuntimeError: Expected all tensors to be on the same device,
            #but found at least two devices, cuda:1 and cuda:2!
            noise = noise.to(predicted_noise.device)#new
            #print("noise devices:",noise.device,predicted_noise.device)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
            #sampled_images = diffusion.sample(model, n=images.shape[0])

        sampled_images = diffusion.sample(model, n=images.shape[0])
        save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional"
    args.epochs = 500
    args.batch_size = 8#16
    args.image_size = 64
    args.dataset_path = 'data'#r"C:\Users\dome\datasets\landscape_img_folder"
    args.device = "cuda:1"
    args.lr = 3e-4
    train(args)


if __name__ == '__main__':
    launch()
