import torch
from torch import nn
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import numpy as np

from gan.config import lr, bs as batch_size
from gan.load_mnist import data_loader
from gan.discriminator import Discriminator
from gan.generator import CNNGenerator as Generator

from datetime import datetime
def now(): return f'{datetime.now():%H:%M@%d-%m-%y}'

generator = Generator().cuda()
discriminator = Discriminator().cuda()

criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)

writer = SummaryWriter(logdir=f'../runs/mnist_gan2_{now()}:cnn2:lr={lr}:bs={batch_size}')

def generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion):
    g_optimizer.zero_grad()
    z = torch.randn(batch_size, 100).cuda()
    fake_images = generator(z)
    validity = discriminator(fake_images)
    g_loss = criterion(validity, torch.ones(batch_size).cuda())
    g_loss.backward()
    g_optimizer.step()
    return g_loss.item()

def discriminator_train_step(batch_size, discriminator, generator, d_optimizer, criterion, real_images, labels):
    d_optimizer.zero_grad()

    # train with real images
    real_validity = discriminator(real_images)
    real_loss = criterion(real_validity, torch.ones(batch_size).cuda())

    # train with fake images
    z = torch.randn(batch_size, 100).cuda()
    fake_images = generator(z)
    fake_validity = discriminator(fake_images)
    fake_loss = criterion(fake_validity, torch.zeros(batch_size).cuda())

    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss.item()

num_epochs = 400
n_critic = 5
display_step = 50
save_step = 1000
for epoch in range(num_epochs):
    # print('Starting epoch {}...'.format(epoch), end=' ')
    for i, (images, labels) in enumerate(data_loader):

        step = epoch * len(data_loader) + i + 1
        real_images = images.cuda()
        labels = labels.cuda()
        generator.train()

        d_loss = discriminator_train_step(len(real_images), discriminator,
                                          generator, d_optimizer, criterion,
                                          real_images, labels)


        for _ in range(n_critic):
            g_loss = generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion)

        writer.add_scalar('cnn_cgan_mnist/g_loss', g_loss, step)
        writer.add_scalar('cnn_cgan_mnist/d_loss', d_loss, step)

        if step % display_step == 0:
            generator.eval()
            z = torch.randn(100, 100).cuda()
            sample_images = generator(z).unsqueeze(1)
            grid = make_grid(sample_images, nrow=10, normalize=True)
            writer.add_image('cnn_cgan_mnist/image', grid, step)
        torch.save(generator.state_dict(), 'models/generator.ptsd')
        print(f'Epoch: {epoch:03d} Step: {i:03d} G-Loss: {g_loss:.3f} D-Loss: {d_loss:.3f}', end='\r')
        writer.flush()
import os; os.makedirs('models/', exist_ok=True)
