import torch
import torch.nn as nn
import torch.nn.functional as F
#model parallel

class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda:0"):
        super().__init__()
        self.inc = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False).to('cuda:2')

    def forward(self, x, t):
        x = x.to('cuda:2')
        print("x dev",x.device)
        x1 = self.inc(x)
        output = x1
        return output