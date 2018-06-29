import torch
from torch import nn
from torch import distributions as d
from torch.nn import functional as F
import numpy as np

# global variables
min_epsilon = 1e-7
max_epsilon = 1. - 1e-7


class VAE(nn.Module):

    def __init__(self, data_dim, hidden_dim, latent_dim, method='Gaussian',
                 variance=1):
        super(VAE, self).__init__()

        # set latent dimension
        self.latent_dim = latent_dim

        # set method: Gaussian, logit-normal, Gumbel-softmax/concrete
        self.method = method
        self._valid_method()

        # set variance for prior in logit
        self.variance = torch.Tensor([variance])

        self.posteriors = None
        self.define_prior()

        # initialize linear layers
        self.input2hidden = nn.Linear(data_dim, hidden_dim)
        self.hidden2hidden_en = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2parameter1 = nn.Linear(hidden_dim, latent_dim)

        # initialize layers for different models:
        if self.method == 'Gaussian' or self.method == 'logit':
            self.hidden2parameter2 = nn.Linear(hidden_dim, latent_dim)
        elif self.method == 'Gumbel':
            self.temperature = torch.Tensor([0.5])

        # initialize activation function
        self.encoder_activation = nn.Hardtanh(min_val=-4.5, max_val=0)

        # initialize layers for decoder:
        self.latent2hidden = nn.Linear(latent_dim, hidden_dim)
        self.hidden2hidden_de = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2output = nn.Linear(hidden_dim, data_dim)

    def _valid_method(self):

        method_unknown_text = "The method is unknown; choose 'Gaussian', " \
            "'logit' or 'Gumbel'."
        assert self.method == 'Gaussian' or self.method == 'logit' or \
            self.method == 'Gumbel', method_unknown_text

    def define_prior(self):
        if self.method == 'Gumbel':
            temperature = torch.Tensor([1.])
            uniform = torch.log(
                1. / torch.Tensor([self.latent_dim])) * \
                torch.ones(self.latent_dim)
            self.prior = d.relaxed_categorical.ExpRelaxedCategorical(
                temperature, logits=uniform)
        else:
            self.prior = None

    def encode(self, x):

        self._valid_method()

        # find hidden layer
        hidden1 = F.tanh(self.input2hidden(x))
        hidden2 = F.softplus(self.hidden2hidden_en(hidden1))

        # find parameters for model of latent variable
        if self.method != 'Gumbel':
            mean = self.hidden2parameter1(hidden2)
            logvar = self.encoder_activation(self.hidden2parameter2(hidden2))
            return (mean, logvar)
        else:
            log_location = self.hidden2parameter1(hidden2)
            return log_location

    @staticmethod
    def create_cov_matrix(logvar, approx):
        var = torch.exp(logvar)
        var_matrix = torch.stack([torch.diag(d) for d in torch.unbind(var)])
        factor_cov_matrix = var_matrix + \
            torch.bmm(approx.unsqueeze(2), approx.unsqueeze(1))
        return factor_cov_matrix

    # we dont use this method on the moment
    def one_hot(self, z):
        indices = z.argmax(dim=1)
        one_hot_z = torch.zeros(z.size())
        for i, index in enumerate(indices.data):
            one_hot_z[i][index] = 1
        return one_hot_z

    def reparameterize(self, parameters):

        self._valid_method()

        if self.method == 'Gaussian':
            (mean, logvar) = parameters
            epsilon = torch.FloatTensor(logvar.size()).normal_()
            std = torch.exp(torch.div(logvar, 2))
            z = mean + epsilon * std

        elif self.method == 'logit':
            (mean, logvar) = parameters
            epsilon = torch.FloatTensor(logvar.size()).normal_()
            std = torch.exp(torch.div(logvar, 2))
            exp_y = torch.exp(mean + epsilon * std)
            z = exp_y / (exp_y + 1)

        elif self.method == 'Gumbel':
            self.posteriors = [d.relaxed_categorical.ExpRelaxedCategorical(
                self.temperature, logits=parameter) for parameter in
                torch.unbind(parameters)]
            z = torch.stack([posterior.rsample()
                             for posterior in self.posteriors])

        return z

    def decode(self, z):

        hidden1 = F.tanh(self.latent2hidden(z))
        hidden2 = F.softplus(self.hidden2hidden_de(hidden1))

        return F.sigmoid(self.hidden2output(hidden2))

    def KL_loss(self, z, parameters):

        log_p_z = self.log_p_z(z)
        log_q_z = self.log_q_z(z, parameters)
        KL_loss = - (log_p_z - log_q_z)

        return KL_loss

    def log_p_z(self, z):

        if self.method == 'Gaussian':
            log_prob = torch.sum(-0.5 * torch.pow(z, 2), 1)

        elif self.method == 'logit':
            variable = torch.log(torch.div(z, 1 - z))
            log_prob = - 0.5 * (self.latent_dim * torch.log(self.variance) +
                                torch.div(torch.sum(torch.pow(variable, 2), 1),
                                          self.variance))

        elif self.method == 'Gumbel':
            log_prob = self.prior.log_prob(z)

        return log_prob

    def log_q_z(self, z, parameters):

        if self.method == 'Gaussian':
            (mean, logvar) = parameters
            log_prob = - 0.5 * \
                (logvar + torch.pow(z - mean, 2) / torch.exp(logvar))

        elif self.method == 'logit':
            variable = torch.log(torch.div(z, 1 - z))
            (mean, logvar) = parameters
            log_prob = -0.5 * \
                (logvar + torch.pow(variable - mean, 2) / torch.exp(logvar))

        elif self.method == 'Gumbel':
            batch_dim = z.size(0)
            z_unbind = torch.unbind(z)
            log_prob = torch.stack([self.posteriors[i].log_prob(
                z_unbind[i]) for i in range(batch_dim)])

        return torch.sum(log_prob, 1)

    def log_bernoulli_loss(self, x, x_mean):
        """
        Negative log Bernoulli loss
        """

        probs = torch.clamp(x_mean, min=min_epsilon, max=max_epsilon)
        loss = - torch.sum(x * torch.log(probs) + (1 - x) *
                           (torch.log(1 - probs)), 1)

        return loss

    def total_loss(self, x, x_mean, z, z_parameters):

        log_bernoulli_loss = self.log_bernoulli_loss(x, x_mean)
        KL_loss = self.KL_loss(z, z_parameters)
        loss = log_bernoulli_loss + KL_loss

        return torch.mean(loss, 0), torch.mean(KL_loss, 0), \
            torch.mean(log_bernoulli_loss, 0)

    def forward(self, x):

        x = x.view(-1, 784)
        parameters = self.encode(x)
        z = self.reparameterize(parameters)
        if self.method == 'logit':
            z = torch.clamp(z, min=min_epsilon, max=max_epsilon)
        x_mean = self.decode(z)

        return x_mean, z, parameters
