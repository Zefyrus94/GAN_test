#si ripete anche in train.py
import torch
from torch import nn
class Discriminator(nn.Module):
    def __init__(self, im_dim=3*64*64, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.make_disc_block(im_dim, hidden_dim * 4),
            self.make_disc_block(hidden_dim * 4, hidden_dim * 2),
            self.make_disc_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def make_disc_block(self, input_dim, output_dim):
        print("input_dim",input_dim,"output_dim", output_dim)
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(0.2,inplace=True)
        )    
    
    def forward(self, image):
        #print("forward",image.shape)
        return self.disc(image)
#
#from torchsummary import summary
#disc = Discriminator().to('cpu') 
#print(disc)
#summary(disc, (3*64*64))
#exit()