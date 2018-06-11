import torch
from torch import nn
from torch import distributions as d
from torch.nn import functional as F 

class VAE(nn.Module):
    
    def __init__(self, data_dim, hidden_dim, latent_dim, method='Gaussian'):
        super(VAE, self).__init__()

        # set method: Gaussian, logit-normal, Gumbel-softmax/concrete
        self.method = method
        self.valid_method()
        
        # initialize first hidden layer for encoder:
        self.input2hidden = nn.Linear(data_dim ,hidden_dim)


        # initialize hidden layers and define activation function:

        if self.method == 'Gaussian' or self.method == 'logit-normal':
        	self.hidden2parameter1 = nn.Linear(hidden_dim, latent_dim)
        	self.hidden2parameter2 = nn.Linear(hidden_dim, latent_dim)
	    	self.encoder_activation = nn.Hardtanh(min_val=-4.5,max_val=0.)

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


    @staticmethod
    def valid_method(self):
    	method_unknown_text = "The method is unknown; choose 'Gaussian', 'logit-normal', 'Gumbel-softmax' or 'concrete'."
        assert self.method == 'Gaussian' or self.method == 'logit-normal' or self.method == 'Gumbel-softmax' or self.method == 'concrete', method_unknown_text


    def encode(self, x):

        x = F.tanh(self.input2hidden(x))
    	parameter1 = self.hidden2parameter1(x)
        parameter2 = self.encoder_activation(self.hidden2parameter2(x))

        return parameter1, parameter2
    

    def reparameterize(self, parameter1, parameter2):

    	self.valid_method()

    	if self.method == 'Gaussian':
    		# parameter1 = mean, parameter2 = logvar
	        epsilon = torch.FloatTensor(parameter2.size()).normal_()
        	std = torch.exp(torch.div(parameter2, 2))
        	z = parameter1 + epsilon * std

	    elif self.method == 'logit-normal':

	    	print("not implemented")
	   #  	epsilon = torch.FloatTensor(parameter2.size()).normal_()
    #     	std = torch.exp(torch.div(parameter2, 2))
    #     	y = parameter1 + epsilon * std
    #     	numerator = torch.exp(y)
    #     	denominator = 1 + torch.sum(numerator, -1)

    # # concat the vector
    #     	z = torch.div(torch.cat((numerator), -1), denominator)

	    elif self.method == 'Gumbel-softmax' or self.method == 'concrete':

	    	print("not implemented")
	    	# parameter1 = log location, parameter2 = temperature/lambda
	    	# gumbel_distr = d.gumbel.Gumbel(parameter1, parameter2)
	    	# epsilon = gumbel_distr.sample(sample_shape=parameter1.size())
	    	# numerator = torch.exp(torch.div(parameter1 + epsilon, parameter2))
	    	# denominator = torch.sum(numerator, -1)
	    	# z = torch.div(numerator, denominator)

        return z


    def decode(self, z):
        z = F.tanh(self.latent2hidden(z))
        return F.sigmoid(self.hidden2output(z))


    def forward(self, x):

        x = x.view(-1, 784)
        parameter1, parameter2 = self.encode(x)
        z = self.reparameterize(parameter1, parameter2)
        output = self.decode(z)
        return output, parameter1, parameter2