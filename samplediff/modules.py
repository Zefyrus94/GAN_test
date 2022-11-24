import torch
import torch.nn as nn
import torch.nn.functional as F
#model sharding
class SelfAttention(nn.Module):
    def __init__(self, channels, size, device='cuda:0'):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        #RuntimeError: Expected all tensors to be on the same device, but found at least two devices, 
        #cuda:0 and cpu! (when checking argument for argument mat2 in method wrapper_mm)
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True).to(device)
        #RuntimeError: Expected all tensors to be on the same device, but found at least two devices, 
        #cuda:3 and cpu! (when checking argument for argument weight in method wrapper__native_layer_norm)
        self.ln = nn.LayerNorm([channels]).to(device)
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]).to(device),
            nn.Linear(channels, channels).to(device),
            nn.GELU(),
            nn.Linear(channels, channels).to(device),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        print("SelfAttention",x_ln.device)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)
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
            DoubleConv(in_channels, in_channels, residual=True, device=device),
            DoubleConv(in_channels, out_channels, device=device),
        )
        #RuntimeError: Expected all tensors to be on the same device, but found at least two devices, 
        #cpu and cuda:1! (when checking argument for argument mat1 in method wrapper_addmm)
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ).to(device),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        print("Down", x.device, t.device)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256, device='cuda:0'):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True, device=device),
            DoubleConv(in_channels, out_channels, in_channels // 2, device=device),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ).to(device),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        print("Up",skip_x.device,x.device)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x

class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda:0"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64, device='cuda:2')
        self.down1 = Down(64, 128, device='cuda:3')
        self.sa1 = SelfAttention(128, 32, device='cuda:0')
        self.down2 = Down(128, 256, device='cuda:1')
        self.sa2 = SelfAttention(256, 16, device='cuda:2')
        self.down3 = Down(256, 256, device='cuda:3')
        self.sa3 = SelfAttention(256, 8, device='cuda:0')

        self.bot1 = DoubleConv(256, 512, device='cuda:1')
        self.bot2 = DoubleConv(512, 512, device='cuda:2')
        self.bot3 = DoubleConv(512, 256, device='cuda:3')

        self.up1 = Up(512, 128, device='cuda:0')
        self.sa4 = SelfAttention(128, 16, device='cuda:1')
        self.up2 = Up(256, 64, device='cuda:2')
        self.sa5 = SelfAttention(64, 32, device='cuda:3')
        self.up3 = Up(128, 64, device='cuda:0')
        self.sa6 = SelfAttention(64, 64, device='cuda:1')
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

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
        x1 = self.inc(x)

        x1 = x1.to('cuda:3')
        t = t.to('cuda:3')
        x2 = self.down1(x1, t)

        x2 = x2.to('cuda:0')
        x2 = self.sa1(x2)

        x2 = x2.to('cuda:1')
        t = t.to('cuda:1')
        x3 = self.down2(x2, t)

        x3 = x3.to('cuda:2')
        x3 = self.sa2(x3)

        x3 = x3.to('cuda:3')
        t = t.to('cuda:3')
        x4 = self.down3(x3, t)

        x4 = x4.to('cuda:0')
        x4 = self.sa3(x4)

        x4 = x4.to('cuda:1')
        x4 = self.bot1(x4)

        x4 = x4.to('cuda:2')
        x4 = self.bot2(x4)

        x4 = x4.to('cuda:3')
        x4 = self.bot3(x4)

        x4 = x4.to('cuda:0')
        x3 = x3.to('cuda:0')
        t = t.to('cuda:0')
        x = self.up1(x4, x3, t)

        x = x.to('cuda:1')
        x = self.sa4(x)

        x = x.to('cuda:2')
        x2 = x2.to('cuda:2')
        t = t.to('cuda:2')
        x = self.up2(x, x2, t)

        x = x.to('cuda:3')
        x = self.sa5(x)

        x = x.to('cuda:0')
        x1 = x1.to('cuda:0')
        t = t.to('cuda:0')
        x = self.up3(x, x1, t)

        x = x.to('cuda:1')
        x = self.sa6(x)

        output = x
        return output