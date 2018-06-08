from torch import nn
from torch.nn import functional as F 


class VAE(nn.Module):
    def __init__(self, fc1_dims, fc21_dims, fc22_dims, fc3_dims, fc4_dims):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(*fc1_dims)
        self.fc21 = nn.Linear(*fc21_dims)
        self.fc22 = nn.Linear(*fc22_dims)
        self.fc3 = nn.Linear(*fc3_dims)
        self.fc4 = nn.Linear(*fc4_dims)

    def encode(self, x):
        embedding = F.relu(self.fc1(x))
        mu = F.sigmoid(self.fc21(embedding))
        logvar = F.tanh(self.fc22(embedding))
        return mu, logvar

    def reparameterize(self, mu, logvar):
        epsilon = torch.normal(torch.zeros(logvar.size()), torch.ones(logvar.size()))
        sigma = torch.sqrt(torch.exp(logvar))
        z = mu + epsilon * logvar
        return z

    def decode(self, z):
        x_hat =  F.sigmoid(self.fc4(self.fc3(z)))
        return x_hat

    def forward(self, x):
        x = x.view(-1, 784)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar