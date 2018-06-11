import torch
from torch.autograd import Variable

# from loss import *
from VAE import *


def train(epoch, train_loader, model, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        optimizer.zero_grad()
        # recon_batch, mean, logvar = model(data)
        recon_batch, z, z_parameter1, z_parameter2 = model(data)
        loss = model.total_loss(data.view(-1, 784), recon_batch, z, z_parameter1, z_parameter2)
        # loss = model.loss_function(recon_batch, data.view(-1, 784), mean, logvar)
        loss.backward()
        train_loss += loss.data
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))