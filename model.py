import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, n_feats=32, depth=3):
        super().__init__()
        
        self.encoder = Encoder(depth=depth, n_feats=n_feats)
        self.decoder = Decoder(depth=depth, n_feats=n_feats)
        self.quantizer = Quantizer(depth=depth, n_feats=n_feats, n_emb=1024, emb_dim=256)

    def forward(self, x):
        feats = self.encoder(x)
        feats, q_loss = self.quantizer(feats)
        rec = self.decoder(feats)
        return rec, feats, q_loss


class Encoder(nn.Module):
    def __init__(self, in_channels=3, n_feats=64, depth=3):
        super().__init__()
        self.input = nn.Conv2d(in_channels, n_feats, 3, 1, 1)

        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(Block(n_feats*2**i, n_feats*2**(i+1), n_feats=n_feats))
            self.layers.append(Downsample(n_feats*2**(i+1)))


    def forward(self, x):
        feats = self.input(x)
        for l in self.layers:
            feats = l(feats)

        return feats

class Decoder(nn.Module):
    def __init__(self, out_channels=3, n_feats=64, depth=4):
        super().__init__()
        self.output = nn.Conv2d(n_feats, out_channels, 3, 1, 1)

        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(Block(n_feats*2**(depth-i), n_feats*2**(depth-i-1), n_feats=n_feats))
            self.layers.append(Upsample(n_feats*2**(depth-i-1)))


    def forward(self, x):
        feats = x
        for l in self.layers:
            feats = l(feats)

        feats = self.output(feats)
        return feats

class Quantizer(nn.Module):
    def __init__(self, depth, n_feats, emb_dim, n_emb):
        super().__init__()
        self.quant_conv = nn.Conv2d(n_feats*2**depth, emb_dim, 1)
        self.quantizer = VectorQuantizer(n_emb, emb_dim)

    def forward(self, feats):
        feats = self.quant_conv(feats)
        return self.quantizer(feats)

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 2, 0)

    def forward(self, x):
        x = torch.nn.functional.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, n_feats=32):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_norm=torch.nn.GroupNorm(num_groups=n_feats, num_channels=self.in_channels, eps=1e-6, affine=True)
        self.in_conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm=torch.nn.GroupNorm(num_groups=n_feats, num_channels=self.out_channels, eps=1e-6, affine=True)
        self.conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

        if(self.in_channels!=self.out_channels):
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        h = self.in_norm(x)
        h=h*torch.sigmoid(h)
        h = self.in_conv(h)
        h = self.norm(h)
        h=h*torch.sigmoid(h)
        h = self.conv(h)
        if(self.in_channels!=self.out_channels):
            x = self.shortcut(x)
        return x+h

class VectorQuantizer(nn.Module):
    def __init__(self, n_emb, emb_dim, beta=0.25):
        super(VectorQuantizer, self).__init__()
        self.n_emb = n_emb
        self.emb_dim = emb_dim
        self.beta = beta

        self.embedding = nn.Embedding(n_emb, emb_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_emb, 1.0 / n_emb)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.emb_dim)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_emb).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)


        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss
