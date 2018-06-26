import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch import nn, optim
from VAE import *
import numpy as np


def train(epoch, train_loader, model, optimizer):
    model.train()
    train_loss = 0
    KL_losses = 0
    log_bernoulli_losses = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        # binarize data
        data = torch.bernoulli(data)
        optimizer.zero_grad()
        recon_batch, z, z_parameters = model(data)
        loss, KL_loss, log_bernoulli_loss = model.total_loss(
            data.view(-1, 784), recon_batch, z, z_parameters)
        loss.backward()
        train_loss += loss.data.numpy()
        KL_losses += KL_loss.data.numpy()
        log_bernoulli_losses += log_bernoulli_loss.data.numpy()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data))

    average_loss = train_loss / len(train_loader)
    average_KL_loss = KL_losses / len(train_loader)
    average_log_bernoulli_loss = log_bernoulli_losses / len(train_loader)

    print("av bernoulli", average_log_bernoulli_loss)
    print("Av KL", average_KL_loss)


    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, average_loss))

    return average_loss, average_KL_loss, average_log_bernoulli_loss, z


def bernoulli(x):
    return np.random.binomial(1, x)


def binarize(data, seed):
    np.random.seed(seed)
    vecbernoulli = np.vectorize(bernoulli)
    data = [(torch.FloatTensor(vecbernoulli(image)), label) for
            (image, label) in data]
    return data


def run_train(latent_dim, epochs, method, train_data, lr, rank1=False, variance=1):

    # set learning rate, batch size and number of epochs
    sample_dim = 784
    hidden_dim = 300
    batch_size = 50

    # Init model
    model = VAE(sample_dim, hidden_dim, latent_dim, method, rank1, variance)

    # Init optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train
    all_losses = []
    KL_losses = []
    log_bernoulli_losses = []
    for epoch in range(1, epochs + 1):
        # binarize data
        seed = np.random.randint(1, 5000)
        torch.manualSeed(seed)
        # Choose one of these for binarization
        # train_data_binary = binarize(train_data, seed)
        train_data_binary = train_data
        train_loader = torch.utils.data.DataLoader(train_data_binary,
                                                   batch_size=batch_size,
                                                   shuffle=True, **{})

        average_loss, average_KL_loss, average_log_bernoulli_loss, z = train(
            epoch, train_loader, model, optimizer)
        all_losses.append(average_loss)
        KL_losses.append(average_KL_loss)
        log_bernoulli_losses.append(average_log_bernoulli_loss)

    return model, all_losses, z, KL_losses, log_bernoulli_losses
