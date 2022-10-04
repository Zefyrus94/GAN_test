#si ripete anche in train.py
import torch
from torch import nn
class Discriminator(nn.Module):
    def __init__(self, im_chan=3, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.make_crit_block(im_chan, hidden_dim),
            self.make_crit_block(hidden_dim, hidden_dim * 2, kernel_size = 3),#ks new
            #
            self.make_crit_block(hidden_dim * 2, hidden_dim * 4, kernel_size = 3),
            #
            self.make_crit_block(hidden_dim * 4, 1, kernel_size = 6, final_layer=True),#hd*2=>1, ks new
        )
    #new padding (unused)
    def make_crit_block(self, input_channels, output_channels, kernel_size=4, stride=2, padding=1,final_layer=False):
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
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)
#
print("preambolo...")
from torchsummary import summary
disc = Discriminator().to('cpu') 
summary(disc, (3,64,64))