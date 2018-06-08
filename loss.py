import torch

min_epsilon = 1e-5
max_epsilon = 1.-1e-5

def log_bernoulli_loss(mean, x):
    probs = torch.clamp(mean, min=min_epsilon, max=max_epsilon)
    loss = torch.sum(x * torch.log(probs) + (1-x) *(torch.log(1-probs)), 1)
    return - torch.sum(loss)

def KL_loss(mu, logvar):
    # Gaussian
    D = torch.FloatTensor([mu.size(1)])
    log_D = torch.log(D)
    sum_logvar = torch.sum(logvar, 1)
    norm_var = torch.sum(torch.exp(logvar), 1)
    norm_mu = torch.sum(mu * mu, 1)
    loss = (log_D - sum_logvar + norm_var + norm_mu - D)/2
    return torch.sum(loss)

def loss_function(x_hat, x, mu, logvar):
    return log_bernoulli_loss(x_hat, x) + KL_loss(mu, logvar)