import torch
from torch import nn
from torch import distributions as d
from torch.nn import functional as F
import numpy as np

# global variables
min_epsilon = 1e-7
max_epsilon = 1. - 1e-7


class VAE(nn.Module):

    def __init__(self, data_dim, hidden_dim, latent_dim, method='Gaussian', rank1=False, variance=1):
        super(VAE, self).__init__()

        # set rank1 (boolean)
        self.rank1 = rank1

        # set latent dimension
        self.latent_dim = latent_dim

        # set method: Gaussian, logit-normal, Gumbel-softmax/concrete
        self.method = method
        self._valid_method()

        # set variance for prior in logit
        self.variance = torch.Tensor([variance])

        # IN CASE BUILT IN MULTIVARIATE FOR RANK1:
        # initialize posterior and prior
        self.posteriors = None
        self.define_prior()     

        # initialize linear layers
        self.input2hidden = nn.Linear(data_dim, hidden_dim)
        self.hidden2hidden_en = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2parameter1 = nn.Linear(hidden_dim, latent_dim)

        # initialize layers for different models:
        if self.method == 'Gaussian' or self.method == 'logit':
            self.hidden2parameter2 = nn.Linear(hidden_dim, latent_dim)
            if self.rank1 and self.method == 'logit':
                self.hidden2parameter3 = nn.Linear(hidden_dim, latent_dim)
        elif self.method == 'Gumbel':
            self.temperature = torch.Tensor([0.5])

        # initialize activation function
        self.encoder_activation = nn.Hardtanh(min_val=-4.5, max_val=0)

        # initialize layers for decoder:
        self.latent2hidden = nn.Linear(latent_dim, hidden_dim)
        self.hidden2hidden_de = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2output = nn.Linear(hidden_dim, data_dim)

    def _valid_method(self):

        method_unknown_text = "The method is unknown; choose 'Gaussian', 'logit', 'logit-sigmoidal' or 'Gumbel'."
        assert self.method == 'Gaussian' or self.method == 'logit' or self.method == 'Gumbel', method_unknown_text

    def define_prior(self):
        if self.method == 'Gumbel':
            temperature = torch.Tensor([1.])
            uniform = torch.log( 1. / torch.Tensor([self.latent_dim]) ) * torch.ones(self.latent_dim)
            self.prior = d.relaxed_categorical.ExpRelaxedCategorical(temperature, logits=uniform)
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
            if self.rank1 and self.method == 'logit':
                # CHECK: Was a sigmoid, changed to tanh (try initilizing with -1)
                approx = F.tanh(self.hidden2parameter3(hidden2))
                return (mean, logvar, approx)
            else:
                return (mean, logvar)
        else:
            # log_location = self.encoder_activation(self.hidden2parameter1(hidden2))
            log_location = self.hidden2parameter1(hidden2)
            return log_location

    @staticmethod
    def create_cov_matrix(logvar, approx):
        var = torch.exp(logvar)
        var_matrix = torch.stack([torch.diag(d) for d in torch.unbind(var)])
        factor_cov_matrix = var_matrix + torch.bmm(approx.unsqueeze(2), approx.unsqueeze(1))
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
            if self.rank1:

                (location, logvar, approx) = parameters

                # THIS IS WITH BUILT IN MULTIVARIATE NORMAL: BACKWARD NOT WORKING
                # batch_dim = location.size(0)
                # location_unbind = torch.unbind(location)
                # Sigma = VAE.create_cov_matrix(logvar, approx)
                # covar = torch.unbind(Sigma)
                # self.posteriors = [d.multivariate_normal.MultivariateNormal(location_unbind[i], covariance_matrix=covar[i]) for i in range(batch_dim)]
                # y = torch.stack([self.posteriors[i].rsample() for i in range(batch_dim)])
                # exp_y = torch.exp(y)
                # z = exp_y / (exp_y + 1)

                # HANDMADE:
                # find cov matrix and factor it
                Sigma = VAE.create_cov_matrix(logvar, approx)
                factor = torch.stack([ torch.potrf(m) for m in torch.unbind(Sigma) ]) # case2: delete this line and replace factor by Sigma
                # sample from standard normal distr
                epsilon = torch.FloatTensor(location.size()).normal_()
                # sample from normal distr N(location, factor)
                exp_y = torch.exp(location + torch.bmm(factor, epsilon.unsqueeze(2)).view(-1, self.latent_dim))
                # compute logit of sample
                z = exp_y / (exp_y + 1)

                # case 2: we first create matrix Sigma= D + aa^T. Then we define cov matrix by Sigma @ Sigma.T

            else:
                (mean, logvar) = parameters
                epsilon = torch.FloatTensor(logvar.size()).normal_()
                std = torch.exp(torch.div(logvar, 2))
                exp_y = torch.exp(mean + epsilon * std)
                z = exp_y / (exp_y + 1)

        elif self.method == 'Gumbel':
            self.posteriors = [d.relaxed_categorical.ExpRelaxedCategorical(self.temperature, logits=parameter) for parameter in torch.unbind(parameters)]
            z = torch.stack([posterior.rsample() for posterior in self.posteriors])

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
            log_prob = torch.sum(-0.5 * torch.pow(z, 2),1)

        elif self.method == 'logit':
            variable = torch.log(torch.div(z, 1 - z))

            # IF BUILT IN MULTIVARIATE NORMAL
            # if self.rank1:
            #     log_prob = self.prior.log_prob(variable)
            # else:
                # ELSE: ONLY DO THIS LINE
            log_prob = - 0.5 * ( self.latent_dim * torch.log(self.variance) + torch.div(torch.sum(torch.pow(variable, 2),1), self.variance) )

        elif self.method == 'Gumbel':
            log_prob = self.prior.log_prob(z)

        return log_prob

    def log_q_z(self, z, parameters):

        if self.method == 'Gaussian':
            (mean, logvar) = parameters
            log_prob = - 0.5 * (logvar + torch.pow(z - mean, 2) / torch.exp(logvar))

        elif self.method == 'logit':
            variable = torch.log(torch.div(z, 1 - z))
            if self.rank1:

                # IF BUILT IN MULTIVARIATE NORMAL
                # batch_dim = z.size(0)
                # z_unbind = torch.unbind(variable)
                # log_prob = torch.stack([self.posteriors[i].log_prob(z_unbind[i]) for i in range(batch_dim)])
                # log_prob = log_prob.unsqueeze(1)

                # HANDMADE:
                (mean, logvar, approx) = parameters
                cov_matrix = VAE.create_cov_matrix(logvar, approx) # call this Sigma in case 2
                # cov_matrix = torch.bmm(Sigma, torch.transpose(Sigma, 1, 2)) # in case 2 (see reparametrize)
                cov_matrix_unbind = torch.unbind(cov_matrix)
                precision_matrix = torch.stack([torch.inverse(m) for m in cov_matrix_unbind])
                log_det_cov_matrix = torch.log(torch.stack([torch.det(m) for m in cov_matrix_unbind]))
                z_minus_mean = variable - mean
                log_prob = - 0.5 * (log_det_cov_matrix + torch.bmm( z_minus_mean.unsqueeze(1), torch.bmm(precision_matrix, z_minus_mean.unsqueeze(2))).view(-1, 1))

            else:
                (mean, logvar) = parameters
                log_prob = -0.5 * (logvar + torch.pow(variable - mean, 2) / torch.exp(logvar))

        elif self.method == 'Gumbel':
            batch_dim = z.size(0)
            z_unbind = torch.unbind(z)
            log_prob = torch.stack([self.posteriors[i].log_prob(z_unbind[i]) for i in range(batch_dim)])

        return torch.sum(log_prob, 1)

    def log_bernoulli_loss(self, x, x_mean):
        """
        Negative log Bernoulli loss
        """

        probs = torch.clamp(x_mean, min=min_epsilon, max=max_epsilon)
        loss = - torch.sum(x * torch.log(probs) + (1 - x) * (torch.log(1 - probs)), 1)

        return loss

    def total_loss(self, x, x_mean, z, z_parameters):

        log_bernoulli_loss = self.log_bernoulli_loss(x, x_mean)
        KL_loss = self.KL_loss(z, z_parameters)
        loss = log_bernoulli_loss + KL_loss

        return torch.mean(loss, 0), torch.mean(KL_loss, 0), torch.mean(log_bernoulli_loss, 0)

    def forward(self, x):

        x = x.view(-1, 784)
        parameters = self.encode(x)
        z = self.reparameterize(parameters)
        if self.method == 'logit':
            z = torch.clamp(z, min=min_epsilon, max=max_epsilon)
        x_mean = self.decode(z)

        return x_mean, z, parameters
