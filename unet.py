import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, channels=128, in_channels=3, depth=4, time_channels=128):
        super().__init__()
        self.time_channels = time_channels
        self.time_input = nn.Sequential(
                nn.Linear(channels, time_channels), 
                nn.SiLU(),
                nn.Linear(time_channels, time_channels)
            )
        self.encoder = Encoder(channels=channels, in_channels=in_channels, depth=depth, time_channels=time_channels)
        mid_channels = channels*2**(depth-1)
        self.mid_blocks = nn.ModuleList([
                ResBlock(mid_channels, time_channels, mid_channels),
                Attention(mid_channels),
                ResBlock(mid_channels, time_channels, mid_channels),
            ])
        self.decoder = Decoder(channels=channels, in_channels=in_channels, depth=depth, time_channels=time_channels)

    def forward(self, x, time=0):
        time = self.time_input(timestep_embedding(time, self.time_channels))
        h, hs = self.encoder(x, time)
        for block in self.mid_blocks:
            h = block(h, time)
        h = self.decoder(h, time, hs)

        return h, hs


class Encoder(nn.Module):
    def __init__(self, channels=128, in_channels=3, depth=3, time_channels=128):
        super().__init__()

        self.input = nn.Conv2d(in_channels, channels, 3, 1, 1)
        self.blocks = nn.ModuleList([])
        ch = channels
        for i in range(depth):
            self.blocks.append(ResBlock(ch, time_channels, channels*2**i, updownsample="down"))
            ch = channels*2**i
            if i >= depth-2:
                self.blocks.append(Attention(channels*2**i))

    def forward(self, x, time=0):
        h = self.input(x)
        hs = []
        for block in self.blocks:
            h = block(h, time)
            hs.append(h)

        return h, hs


class Decoder(nn.Module):
    def __init__(self, channels=128, in_channels=3, depth=3, time_channels=128):
        super().__init__()

        self.blocks = nn.ModuleList([])
        ch = channels
        for i in range(depth):
            self.blocks.append(ResBlock(2*channels*2**i, time_channels, ch, updownsample="up"))
            ch = channels*2**i
            if i >= depth-2:
                self.blocks.append(Attention(channels*2**i))

        self.output = nn.Conv2d(channels, in_channels, 3, 1, 1)

    def forward(self, h, time, hs):
        for block, old_h in reversed(list(zip(self.blocks, hs))):
            if(isinstance(block, ResBlock)):
                h = torch.cat([h, old_h], dim=1)
            h = block(h, time)

        h = self.output(h)
        return h


class ResBlock(nn.Module):
    def __init__(self, channels, emb_channels, out_channels, updownsample="none"):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels

        self.in_layers = nn.Sequential(
                nn.GroupNorm(32, channels),
                nn.SiLU(),
            )

        if updownsample == "up":
            self.updownsample = nn.Upsample(scale_factor=2, mode="nearest")
        elif updownsample == "down":
            self.updownsample = nn.AvgPool2d(2, 2)
        else:
            self.updownsample = nn.Identity()

        self.in_conv = nn.Conv2d(channels, out_channels, 3, 1, 1)

        self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_channels, out_channels)
            )

        self.out_layers = nn.Sequential(
                nn.GroupNorm(32, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1)
            )

        self.skip_connection = nn.Identity() if channels==out_channels else nn.Conv2d(channels, out_channels, 1)

    def forward(self, x, time=0):
        emb = self.emb_layers(time).unsqueeze(-1).unsqueeze(-1)
        h = self.in_layers(x)
        x = self.updownsample(x)
        h = self.updownsample(h)
        h = self.in_conv(h)
        h = h + emb
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels*3, 1)
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x, *args):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x)).permute(0,2,1)
        q, k, v = qkv.chunk(3, dim=-1)
        attention, _ = self.mha(q,k,v)
        attention = self.proj(attention.permute(0, 2, 1))
        return (x+attention).reshape(b, c, *spatial)


def timestep_embedding(timesteps, channels, max_period=10000):
    half = channels// 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding
