import torch
from torch import nn
from torch import distributions as d
from torch.nn import functional as F
import numpy as np

# global variables
min_epsilon = 1e-7
max_epsilon = 1.-1e-7

class VAE(nn.Module):
    

    def __init__(self, data_dim, hidden_dim, latent_dim, method='Gaussian', rank1=False):
        super(VAE, self).__init__()

        # set rank1 (boolean)
        self.rank1 = rank1

        # set latent dimension
        self.latent_dim = latent_dim

        # set method: Gaussian, logit-normal, Gumbel-softmax/concrete
        self.method = method
        self._valid_method()
        
        # initialize first hidden layer for encoder:
        self.input2hidden = nn.Linear(data_dim ,hidden_dim)

        # initialize hidden layers and define activation function:

        if self.method == 'Gaussian' or self.method == 'logit':
            self.hidden2parameter1 = nn.Linear(hidden_dim, latent_dim)
            self.hidden2parameter2 = nn.Linear(hidden_dim, latent_dim)
            if self.rank1 and self.method == 'logit':
                self.hidden2parameter3 = nn.Linear(hidden_dim, latent_dim)
            self.encoder_activation = nn.Hardtanh(min_val=-4.5,max_val=0)

        elif self.method == 'logit-sigmoidal':
            self.hidden2parameter1 = nn.Linear(hidden_dim, latent_dim-1)
            self.hidden2parameter2 = nn.Linear(hidden_dim, latent_dim-1)
            self.encoder_activation = nn.Hardtanh(min_val=-4.5,max_val=0.)

        elif self.method == 'Gumbel':
            self.hidden2parameter1 = nn.Linear(hidden_dim, latent_dim)
            self.temperature = torch.Tensor([10])

        # initialize layers for decoder:
        self.latent2hidden = nn.Linear(latent_dim, hidden_dim)
        self.hidden2output = nn.Linear(hidden_dim, data_dim)


    def _valid_method(self):

        method_unknown_text = "The method is unknown; choose 'Gaussian', 'logit', 'logit-sigmoidal' or 'Gumbel'."
        assert self.method == 'Gaussian' or self.method == 'logit' or self.method == 'logit-sigmoidal' or self.method == 'Gumbel', method_unknown_text


    def encode(self, x):

        self._valid_method()

        # find hidden layer
        hidden = F.tanh(self.input2hidden(x))

        # find parameters for model of latent variable
        if self.method != 'Gumbel':
            mean = self.hidden2parameter1(hidden)
            if self.rank1 and self.method == 'logit':
                logvar = self.encoder_activation(self.hidden2parameter2(hidden))
                approx = F.sigmoid(self.hidden2parameter3(hidden))
                return (mean, logvar, approx)
            else:
                logvar = self.encoder_activation(self.hidden2parameter2(hidden))
                return (mean, logvar)
        else:            
            log_location =  F.logsigmoid(self.hidden2parameter1(hidden))
            return log_location


    @staticmethod
    def create_factor_cov_matrix(logvar, approx):

        var = torch.exp(logvar)
        var_matrix = torch.stack([ torch.diag(d) for d in torch.unbind(var) ])
        factor_cov_matrix = var_matrix + torch.bmm( approx.unsqueeze(2), approx.unsqueeze(1) )

        return factor_cov_matrix


    def reparameterize(self, parameters):

        self._valid_method()

        if self.method == 'Gaussian':
            (mean, logvar) = parameters
            epsilon = torch.FloatTensor(logvar.size()).normal_()
            std = torch.exp(torch.div(logvar, 2))
            z = mean + epsilon * std

        elif self.method == 'logit':
            if self.rank1:
                (mean, logvar, approx) = parameters
                epsilon = torch.FloatTensor(mean.size()).normal_()
                Sigma = VAE.create_factor_cov_matrix(logvar, approx)
                # cov_matrix = torch.stack([ torch.inverse(m) for m in torch.unbind(VAE.create_inverse_cov_matrix(logvar, approx)) ])
                # computes the Cholesky decomposition cov_matrix = Sigma.T @ Sigma = Sigma @ Sigma.T
                # Sigma = torch.stack([ torch.potrf(m) for m in cov_matrix ])
                exp_y = torch.exp(mean + torch.bmm(Sigma, epsilon.unsqueeze(2)).view(-1, self.latent_dim))
                z = exp_y / ( exp_y + 1 )
            else:
                (mean, logvar) = parameters
                epsilon = torch.FloatTensor(logvar.size()).normal_()
                std = torch.exp(torch.div(logvar, 2))
                exp_y = torch.exp(mean + epsilon * std)
                z = exp_y / ( exp_y + 1 )

        elif self.method == 'logit-sigmoidal':
            (mean, logvar) = parameters
            epsilon = torch.FloatTensor(logvar.size()).normal_()
            std = torch.exp(torch.div(logvar, 2))
            y = mean + epsilon * std
            numerator = torch.exp(y)
            denominator = 1 + torch.sum(numerator, 1).unsqueeze(1)
            z = torch.div(torch.cat((numerator, torch.ones((numerator.size(0),1))), 1), denominator)

        elif self.method == 'Gumbel':
            log_location = parameters
            gumbel_distr = d.gumbel.Gumbel(0, 1)
            epsilon = gumbel_distr.sample(sample_shape=log_location.size())
            numerator = torch.exp(torch.div(log_location + epsilon, self.temperature))
            # denominator = torch.sum(numerator, 1).unsqueeze(1)
            # z = torch.div(numerator, denominator)
            z = numerator

        return z


    def decode(self, z):

        z = F.tanh(self.latent2hidden(z))

        return F.sigmoid(self.hidden2output(z))


    def KL_loss(self, z, parameters):

        log_p_z = self.log_p_z(z)
        log_q_z = self.log_q_z(z, parameters)
        KL_loss = - ( log_p_z - log_q_z )

        return KL_loss


    def log_p_z(self, z):

        if self.method == 'Gaussian':
            log_distr = -0.5 * torch.pow(z, 2)

        elif self.method == 'logit':
            variable = torch.log(torch.div( z, 1-z ))
            log_distr = -0.5 * torch.pow( variable, 2 )

        elif self.method == 'logit-sigmoidal':
            variable = torch.log(torch.div(z[:, :-1], z[:,-1].unsqueeze(1)))
            log_distr = -0.5 * torch.pow(variable, 2 )

        elif self.method == 'Gumbel':
            k = torch.Tensor([z.size(1)])
            log_location = torch.log( 1 / k )
            log_z = torch.log(z)
            logsumexp = torch.log( torch.sum( torch.exp( log_location - log_z ) , 1 ) )
            log_distr = k - 1 - k * logsumexp + torch.sum( log_location - 2 * log_z , 1 )
            log_distr = log_distr.unsqueeze(1)

        return torch.sum(log_distr, 1)


    def log_q_z(self, z, parameters):

        if self.method == 'Gaussian':
            (mean, logvar) = parameters
            log_distr = - 0.5 * ( logvar + torch.pow(z - mean, 2) / torch.exp(logvar) )

        elif self.method == 'logit':
            variable = torch.log(torch.div( z, 1-z ))
            if self.rank1:
                (mean, logvar, approx) = parameters
                Sigma = VAE.create_factor_cov_matrix(logvar, approx)
                cov_matrix = torch.bmm(Sigma, torch.transpose(Sigma, 1, 2))
                inverse_cov_matrix = torch.stack([ torch.inverse(m) for m in torch.unbind(cov_matrix) ])
                log_det_cov_matrix = torch.log(torch.stack([ torch.det(m) for m in torch.unbind(inverse_cov_matrix) ])) 
                z_minus_mean = variable - mean
                log_distr = 0.5 * ( log_det_cov_matrix - torch.bmm(z_minus_mean.unsqueeze(1), torch.bmm( inverse_cov_matrix, z_minus_mean.unsqueeze(2) ) ).view(-1, 1) )
            else:
                (mean, logvar) = parameters
                log_distr = -0.5 * ( logvar + torch.pow(variable - mean, 2) / torch.exp(logvar) )

        elif self.method == 'logit-sigmoidal':
            (mean, logvar) = parameters
            variable = torch.log(torch.div(z[:, :-1], z[:,-1].unsqueeze(1)))
            log_distr = -0.5 * ( logvar + torch.pow(variable - mean, 2) / torch.exp(logvar) )

        elif self.method == 'Gumbel':
            log_location = parameters
            log_z = torch.log(z)
            logsumexp = torch.log(torch.sum(torch.exp(log_location - self.temperature * log_z ), 1) )
            k = torch.Tensor([z.size(1)])
            log_distr = (k-1) * torch.log(self.temperature) - k * logsumexp + torch.sum(log_location + (self.temperature + 1) * log_z, 1)
            log_distr = log_distr.unsqueeze(1)
            
        return torch.sum(log_distr, 1)


    def log_bernoulli_loss(self, x, x_mean):
        """ 
        Negative log Bernoulli loss

        """
        probs = torch.clamp(x_mean, min=min_epsilon, max=max_epsilon)
        loss = torch.sum(x * torch.log(probs) + (1-x) *(torch.log(1-probs)), 1)

        return - torch.sum(loss)
    

    def total_loss(self, x, x_mean, z, z_parameters):

        log_bernoulli_loss = self.log_bernoulli_loss(x, x_mean)
        KL_loss = self.KL_loss(z, z_parameters)
        loss = log_bernoulli_loss + KL_loss

        # do we want to take the sum or mean???
        return torch.mean(loss, 0)


    def forward(self, x):

        x = x.view(-1, 784)
        parameters = self.encode(x)
        z = self.reparameterize(parameters)
        if self.method == 'logit' or self.method == 'logit-sigmoidal':
            z = torch.clamp(z, min=min_epsilon, max=max_epsilon)
        elif self.method == 'Gumbel':
            z = torch.clamp(z, min=min_epsilon)
        x_mean = self.decode(z)

        return x_mean, z, parameters

