import torch

from torch.autograd import backward, grad

from torch.distributions import Dirichlet as Dir
from torch.distributions.kl import kl_divergence as kl_div

dim = 20
alpha = torch.ones(dim)
beta = torch.ones(dim)

true = 4 * torch.randn(dim)


