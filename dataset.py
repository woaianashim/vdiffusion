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
                transforms.Resize((64, 64)),
                transforms.ToTensor()
            ])
        super().__init__(os.path.join(path, "flower"), download=True, transform=transform)


class NoiseFlower(Flowers102):
    def __init__(self, path):
        transform = transforms.Compose([
                transforms.CenterCrop(500),
                transforms.Resize((64, 64)),
                transforms.ToTensor()
            ])
        super().__init__(os.path.join(path, "flower"), download=True, transform=transform)

    def __getitem__(self, index, time=None, noise_steps=100, beta_init=1e-4, beta_final=2e-2):
        if not time:
            time = random.randint(1, noise_steps)

        image = super().__getitem__(index)[0]
        betas = torch.linspace(beta_init, beta_final, noise_steps)
        alphas = 1. - betas[:time]
        alphas_hat = alphas.prod()

        epsilon = torch.randn(*image.shape)

        return torch.sqrt(alphas_hat) * image + torch.sqrt(1 - alphas_hat) * epsilon, epsilon, image, time

        


