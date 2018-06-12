import torch
from torch import nn
from torch import distributions as d
from torch.nn import functional as F
import numpy as np

# global variables
min_epsilon = 1e-5
max_epsilon = 1.-1e-5

class VAE(nn.Module):
    

    def __init__(self, data_dim, hidden_dim, latent_dim, method='Gaussian'):
        super(VAE, self).__init__()

        # set method: Gaussian, logit-normal, Gumbel-softmax/concrete
        self.method = method
        self._valid_method()
        
        # initialize first hidden layer for encoder:
        self.input2hidden = nn.Linear(data_dim ,hidden_dim)


        # initialize hidden layers and define activation function:

        if self.method == 'Gaussian':
            self.hidden2parameter1 = nn.Linear(hidden_dim, latent_dim)
            self.hidden2parameter2 = nn.Linear(hidden_dim, latent_dim)
            self.encoder_activation = nn.Hardtanh(min_val=-4.5,max_val=0)

        elif self.method == 'logit-normal':
            self.hidden2parameter1 = nn.Linear(hidden_dim, latent_dim-1)
            self.hidden2parameter2 = nn.Linear(hidden_dim, latent_dim-1)
            self.encoder_activation = nn.Hardtanh(min_val=-4.5,max_val=0.)

        elif self.method == 'Gumbel-softmax' or self.method == 'concrete':
            self.hidden2parameter1 = nn.Linear(hidden_dim, latent_dim)
            self.hidden2parameter2 = nn.Linear(hidden_dim, 1)
            # lambda > 0
            self.encoder_activation = nn.Sigmoid()

        # initialize layers for decoder:
        self.latent2hidden = nn.Linear(latent_dim, hidden_dim)
        self.hidden2output = nn.Linear(hidden_dim, data_dim)


    def _valid_method(self):
        method_unknown_text = "The method is unknown; choose 'Gaussian', 'logit-normal', 'Gumbel-softmax' or 'concrete'."
        assert self.method == 'Gaussian' or self.method == 'logit-normal' or self.method == 'Gumbel-softmax' or self.method == 'concrete', method_unknown_text


    def encode(self, x):

        x = F.tanh(self.input2hidden(x))
        parameter1 = self.hidden2parameter1(x)
        parameter2 = self.encoder_activation(self.hidden2parameter2(x))

        return parameter1, parameter2
    

    def reparameterize(self, parameter1, parameter2):

        # repa built in
        # gumbel softmax

        self._valid_method()

        if self.method == 'Gaussian':
            # parameter1 = mean, parameter2 = logvar
            epsilon = torch.FloatTensor(parameter2.size()).normal_()
            std = torch.exp(torch.div(parameter2, 2))
            z = parameter1 + epsilon * std

        elif self.method == 'logit-normal':
            epsilon = torch.FloatTensor(parameter2.size()).normal_()
            std = torch.exp(torch.div(parameter2, 2))
            y = parameter1 + epsilon * std
            numerator = torch.exp(y)
            denominator = 1 + torch.sum(numerator, 1).unsqueeze(1)
            z = torch.div(torch.cat((numerator, torch.ones((numerator.size(0),1))), 1), denominator)

        elif self.method == 'Gumbel-softmax' or self.method == 'concrete':
            # parameter1 = log location, parameter2 = temperature/lambda
            gumbel_distr = d.gumbel.Gumbel(0, 1)
            epsilon = gumbel_distr.sample(sample_shape=parameter1.size())
            numerator = torch.exp(torch.div(parameter1 + epsilon, parameter2))
            denominator = torch.sum(numerator, 1).unsqueeze(1)
            z = torch.div(numerator, denominator)

        return z


    def decode(self, z):
        z = F.tanh(self.latent2hidden(z))
        return F.sigmoid(self.hidden2output(z))


    def KL_loss(self, z, parameter1, parameter2):

        log_p_z = self.log_p_z(z)
        log_q_z = self.log_q_z(z, parameter1, parameter2)
        KL_loss = -(log_p_z - log_q_z)

        return KL_loss


    def log_p_z(self, z):

        if self.method == 'Gaussian':
            log_distr = -0.5 * torch.pow(z, 2)

        elif self.method == 'logit-normal':
            variable = torch.log(torch.div(z[:, :-1], z[:,-1].unsqueeze(1)))
            log_distr = -0.5 * torch.pow(variable, 2 )

        elif self.method == 'Gumbel-softmax' or self.method == 'concrete':
            log_distr = torch.exp(-torch.exp(-z))

        return torch.sum(log_distr, 1)


    def log_q_z(self, z, parameter1, parameter2):

        if self.method == 'Gaussian':
            # cov matrix is a diag matrix
            # param 1 = mean
            # param 2 = logvar
            log_distr = -0.5 * ( parameter2 + torch.pow(z - parameter1, 2) / torch.exp(parameter2) )

        elif self.method == 'logit-normal':
            variable = torch.log(torch.div(z[:, :-1], z[:,-1].unsqueeze(1)))
            log_distr = -0.5 * ( parameter2 + torch.pow(variable - parameter1, 2) / torch.exp(parameter2) )

        elif self.method == 'Gumbel-softmax' or self.method == 'concrete':
            variable = ( z - parameter1 )/ parameter2
            log_distr = torch.exp((-variable+ torch.exp(-variable)))/ parameter2
            # gumbel(param1, param2)

        return torch.sum(log_distr, 1)


    def log_bernoulli_loss(self, x, x_mean):
        """ 
        Negative log Bernoulli loss

        """
        probs = torch.clamp(x_mean, min=min_epsilon, max=max_epsilon)
        loss = torch.sum(x * torch.log(probs) + (1-x) *(torch.log(1-probs)), 1)

        return - torch.sum(loss)

    
    

    def total_loss(self, x, x_mean, z, z_parameter1, z_parameter2):

        log_bernoulli_loss = self.log_bernoulli_loss(x, x_mean)
        KL_loss = self.KL_loss(z, z_parameter1, z_parameter2)
        loss = log_bernoulli_loss + KL_loss

        # do we want to take the sum or mean???
        return torch.mean(loss, 0)


    def forward(self, x):

        x = x.view(-1, 784)
        z_parameter1, z_parameter2 = self.encode(x)
        z = self.reparameterize(z_parameter1, z_parameter2)
        x_mean = self.decode(z)

        return x_mean, z, z_parameter1, z_parameter2