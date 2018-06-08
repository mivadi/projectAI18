import torch
from torch import nn
from torch.nn import functional as F 


class VAE(nn.Module):
    
    def __init__(self, sample_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        
        # initialize layers for encoder:
        self.input2hidden = nn.Linear(sample_dim ,hidden_dim)
        self.hidden2mean = nn.Linear(hidden_dim, latent_dim)
        self.hidden2logvar = nn.Linear(hidden_dim, latent_dim)
        
        # initialize layers for decoder:
        self.latent2hidden = nn.Linear(latent_dim, hidden_dim)
        self.hidden2output = nn.Linear(hidden_dim, sample_dim)

    def encode(self, x):
        x = F.tanh(self.input2hidden(x))
        mean = self.hidden2mean(x)
        logvar = self.hidden2logvar(x)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        epsilon = torch.FloatTensor(logvar.size()).normal_()
        std = torch.exp(torch.div(logvar, 2))
        z = mean + epsilon * std
        return z

    def decode(self, z):
        z = F.tanh(self.latent2hidden(z))
        return F.sigmoid(self.hidden2output(z))

    def forward(self, x):
        x = x.view(-1, 784)
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        output = self.decode(z)
        return output, mean, logvar