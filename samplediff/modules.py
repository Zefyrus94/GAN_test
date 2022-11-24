import torch
import torch.nn as nn
import torch.nn.functional as F
#model parallel
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return self.double_conv(x)
class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda:0"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        #senza .to('cuda:2'): RuntimeError:
        #Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
        self.inc = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False).to('cuda:2')
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        #print("pos_encoding dev t",t.device.index," inv_freq,",inv_freq.device.index)
        print("pos_encoding",t.shape)
        print(t,inv_freq)
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        #t = t.to('cuda:2')
        x = x.to('cuda:2')
        print("x dev",x.device)
        x1 = self.inc(x)
        output = x1
        return output