#si ripete anche in train.py
import torch
from torch import nn
class Generator(nn.Module):
    def __init__(self, input_dim=10, im_dim=3*64*64, hidden_dim=128):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(input_dim, hidden_dim),
            self.make_gen_block(hidden_dim, hidden_dim * 2),
            self.make_gen_block(hidden_dim * 2, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid()
        )
    def make_gen_block(self, input_dim, output_dim, final=False):
        #if final:
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )    
    def forward(self, noise):
        return self.gen(noise)
#
#from torchsummary import summary
#gen = Generator(64).to('cpu')
#summary(gen, (64,))
