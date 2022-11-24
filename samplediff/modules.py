import torch
import torch.nn as nn
import torch.nn.functional as F
#model parallel
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False, device='cuda:0'):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        #senza .to(device): RuntimeError:
        #Conv2d: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
        #GroupNorm Expected all tensors to be on the same device, but found at least two devices, cuda:2 and cpu!
        # (when checking argument for argument weight in method wrapper__native_group_norm)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False).to(device),
            nn.GroupNorm(1, mid_channels).to(device),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False).to(device),
            nn.GroupNorm(1, out_channels).to(device),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256, device="cuda:0"):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        return x
        
class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda:0"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64, device='cuda:2')
        self.down1 = Down(64, 128, device='cuda:3')
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
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
        x1 = x1.to('cuda:3')
        x2 = self.down1(x1, t)
        output = x2
        return output