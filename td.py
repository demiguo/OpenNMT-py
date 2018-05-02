import torch

from torch.autograd import backward, grad

from torch.distributions import Dirichlet as Dir
from torch.distributions.kl import kl_divergence as kl_div

import torch.nn.functional as F

lr = 0.01

dim = 50
alpha = torch.ones(dim, requires_grad=True)
beta = torch.randn(dim, requires_grad=True)
beta.data.mul_(2).exp_()

true = Dir(4 * torch.randn(dim))

data = true.rsample(torch.Size([256]))

print(alpha)
print(beta)

# GD
for i in range(20):
    kl = kl_div(Dir(alpha), Dir(beta))
    print("KL div: {}".format(kl.item()))
    kl.backward()
    print("alpha: {}".format(alpha))
    #print("alpha.grad: {}".format(alpha.grad))
    print("beta: {}".format(beta))
    #print("beta.grad: {}".format(beta.grad))

    alpha.data = alpha.data - lr * alpha.grad.data
    beta.data = beta.data - lr * beta.grad.data
    #import pdb; pdb.set_trace()

# Newton
for i in range(20):
    kl = kl_div(Dir(alpha), Dir(beta))
    print("KL div: {}".format(kl.item()))

    import pdb; pdb.set_trace()
    print("alpha: {}".format(alpha))
    #print("alpha.grad: {}".format(alpha.grad))
    print("beta: {}".format(beta))
    #print("beta.grad: {}".format(beta.grad))

    alpha.data = alpha.data - lr * alpha.grad.data
    beta.data = beta.data - lr * beta.grad.data
    #import pdb; pdb.set_trace()

