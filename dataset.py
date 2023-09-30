import os
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
