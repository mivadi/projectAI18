import torch
from torch import nn
from torch.nn import functional as F 

# encoder -> make options for gaussian/ logit etc.
# decoder -> binary -> look into given code
# implement the gumbel softmax

class VAE(nn.Module):
    
    def __init__(self, sample_dim):
        super(VAE, self).__init__()

        # encoder:
        self.input2hidden = nn.Linear(sample_dim ,300)
        self.hidden2mean = nn.Linear(300, 300)
        self.hidden2logvar = nn.Linear(300, 300)
        
        # decoder (discrete/binary output):
        self.latent2hidden = nn.Linear(300, 300)
        self.hidden2output = nn.Linear(300, sample_dim)
        # for a continuous output -> make two hidden2out (one for mean and one for covariance)

    def encode(self, x):
        x = F.tanh(self.input2hidden(x))
        mu = self.hidden2mean(x)
        logvar = self.hidden2logvar(x)
        return mu, logvar

    # this is what they did in their git
    # def reparameterize(self, mu, logvar):
    #         std = logvar.mul(0.5).exp_()
    #         if self.args.cuda:
    #             eps = torch.cuda.FloatTensor(std.size()).normal_()
    #         else:
    #             eps = torch.FloatTensor(std.size()).normal_()
    #         eps = Variable(eps)
    #         return eps.mul(std).add_(mu)
    
    def reparameterize(self, mu, logvar):
#         return mu + torch.exp(torch.div(logvar, 2)) * torch.randn_like(logvar)
        epsilon = torch.normal(torch.zeros(logvar.size()), torch.ones(logvar.size()))
        sigma = torch.exp(torch.div(logvar, 2))
        z = mu + epsilon * logvar
        return z

    def decode(self, z):
        z = F.tanh(self.latent2hidden(z))
        return F.sigmoid(self.hidden2output(z))

    def forward(self, x):
        x = x.view(-1, 784)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar