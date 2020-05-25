import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.label_emb = nn.Embedding(10, 10)

        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        z = z.view(z.size(0), 100)
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out.view(x.size(0), 28, 28)

# ---✂✂✂---------------------------------------------------------------
# ---CNN GENERATOR-----------------------------------------------------
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class OldCNNGenerator(nn.Module):
    def __init__(self, emb_sz=32):
        super().__init__()
        self.emb_sz = emb_sz
        self.label_emb = nn.Embedding(10, self.emb_sz)

        self.model = nn.Sequential(
            nn.Linear(100+self.emb_sz, 256),
            nn.LeakyReLU(0.2),
            Reshape(-1,1,16,16),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(1, 128, (3,3), dilation=2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 256, (3,3), dilation=2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 1, (3,3), dilation=2),
            nn.Tanh()
        )

    def forward(self, z, labels):
        z = z.view(z.size(0), 100)
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out.view(x.size(0), 28, 28)

class CNNGenerator(nn.Module):
    def __init__(self, emb_sz=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 7*7*128),
            nn.LeakyReLU(0.2),
            Reshape(-1,128,7,7),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 128, (2,2), stride=2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 1, (2,2), stride=2),
            nn.Tanh()
        )
        # summary(model, input_size=(132,), device='cpu')

    def forward(self, z):
        out = self.model(z)
        return out.view(z.size(0), 28, 28)
