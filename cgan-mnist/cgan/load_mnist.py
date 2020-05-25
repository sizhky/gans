import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from .config import bs as batch_size

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
])

data_loader = torch.utils.data.DataLoader(MNIST('~/data', train=True, download=True, transform=transform),
                                          batch_size=batch_size, shuffle=True, drop_last=True)
