import os
import torch
import random
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import Flowers102

class Flower(Flowers102):
    def __init__(self, path):
        transform = transforms.Compose([
                transforms.CenterCrop(500),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        super().__init__(os.path.join(path, "flower"), download=True, transform=transform, split="test")


class NoiseFlower(Flowers102):
    def __init__(self, path, noise_steps=100, beta_init=1e-4, beta_final=2e-2):
        transform = transforms.Compose([
                transforms.CenterCrop(500),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        super().__init__(os.path.join(path, "flower"), download=True, transform=transform)

        betas = torch.linspace(beta_init, beta_final, noise_steps)
        alphas = 1. - betas
        self.alphas_hat = alphas.cumprod(0)
        self.noise_steps = noise_steps

    def __getitem__(self, index, time=None, ):
        if not time:
            time = random.randint(0, self.noise_steps-1)


        image = super().__getitem__(index)[0]
        alpha_hat = self.alphas_hat[time]
        epsilon = torch.randn(*image.shape)

        return torch.sqrt(alpha_hat) * image + torch.sqrt(1 - alpha_hat) * epsilon, epsilon, image, torch.tensor(time)

class Converted(Dataset):
    def __init__(self, path):
        self.path = path
        self.names = os.listdir(path)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        feats = torch.load(os.path.join(self.path, self.names[index]), map_location="cpu")
        return feats

        
class NoiseConverted(Converted):
    def __init__(self, path, noise_steps=100, beta_init=1e-4, beta_final=2e-2):
        super().__init__(os.path.join(path, "converted"))

        betas = torch.linspace(beta_init, beta_final, noise_steps)
        alphas = 1. - betas
        self.alphas_hat = alphas.cumprod(0)
        self.noise_steps = noise_steps

    def __getitem__(self, index, time=None, ):
        if not time:
            time = random.randint(0, self.noise_steps-1)


        image = super().__getitem__(index)
        alpha_hat = self.alphas_hat[time]
        epsilon = torch.randn(*image.shape)

        return torch.sqrt(alpha_hat) * image + torch.sqrt(1 - alpha_hat) * epsilon, epsilon, image, torch.tensor(time)


