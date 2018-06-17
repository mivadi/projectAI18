import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch import nn, optim
from VAE import *
import numpy as np

def train(epoch, train_loader, model, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        optimizer.zero_grad()
        recon_batch, z, z_parameters = model(data)
        loss = model.total_loss(data.view(-1, 784), recon_batch, z, z_parameters)
        loss.backward()
        train_loss += loss.data
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data / len(data)))

    average_loss = train_loss / len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format( epoch, average_loss))

    return average_loss


def run_train(latent_dim, epochs, method, train_loader, lr):
    # set learning rate, batch size and number of epochs
    sample_dim = 784
    hidden_dim = 300

    # Init model
    model = VAE(sample_dim, hidden_dim, latent_dim, method)

    # Init optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train
    all_losses = []
    for epoch in range(1, epochs + 1):
        average_loss = train(epoch, train_loader, model, optimizer)
        all_losses.append(average_loss)

    return model, all_losses