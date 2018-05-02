import torch

from torch.autograd import backward, grad

from torch.distributions import Dirichlet as Dir
from torch.distributions.kl import kl_divergence as kl_div

import torch.nn.functional as F

steps = 10
lr = 0.01

dim = 20
alpha = torch.randn(dim, requires_grad=True)
alpha.data.mul_(2)
beta = torch.randn(dim, requires_grad=True)
beta.data.mul_(2)

true = Dir(4 * torch.randn(dim))
data = true.rsample(torch.Size([256]))

print(alpha.data.clamp(-2, 5).exp())
print(beta.data.clamp(-2, 5).exp())

# GD
def gd(alpha, beta):
    print("GRADIENT DESCENT")
    alpha, beta = alpha.detach(), beta.detach()
    alpha.requires_grad = True
    beta.requires_grad = True
    for i in range(steps):
        print("Step {}".format(i))
        kl = kl_div(Dir(alpha.clamp(-2, 5).exp()), Dir(beta.clamp(-2, 5).exp()))
        print("KL div: {}".format(kl.item()))
        kl.backward()
        print("ealpha: {}".format(alpha.data.clamp(-2, 5).exp()))
        #print("alpha.grad: {}".format(alpha.grad))
        print("ebeta: {}".format(beta.data.clamp(-2, 5).exp()))
        #print("beta.grad: {}".format(beta.grad))

        alpha.data = alpha.data - lr * alpha.grad.data
        beta.data = beta.data - lr * beta.grad.data
        alpha.grad.data.zero_()
        beta.grad.data.zero_()

def nat(alpha, beta):
    print("Natural GD")
    alpha, beta = alpha.detach(), beta.detach()
    alpha.requires_grad = True
    beta.requires_grad = True

    def dir_natural_gradient(grad, alpha):
        # see Finv, this is the batched version
        N, T, S = alpha.size()
        res = alpha.new(N*T, S, S).fill_(0)
        res.as_strided(torch.Size([N*T, S]), [res.stride(0), res.size(2)+1]) \
            .copy_(torch.polygamma(1, alpha.view(N*T, S)))
        Fim = res - torch.polygamma(1, alpha.sum())
        Finv = torch.cat([x.squeeze(0).inverse().unsqueeze(0) for x in Fim.split(1)], 0)
        return torch.bmm(Finv, grad.view(N*T, S, 1)).view(N, T, S)

    # Newton
    def Finv(theta):
        d = torch.diag(torch.polygamma(1, theta))
        o = torch.polygamma(1, theta.sum())
        return (d-o).inverse()

    for i in range(steps):
        print("Step {}".format(i))
        ealpha = alpha.clamp(-2, 5).exp()
        ebeta = beta.clamp(-2, 5).exp()

        def ealpha_hook(grad):
            # see Finv, this is the batched version
            N, T, S = 1, 1, dim
            res = ealpha.new(N*T, S, S).fill_(0)
            res.as_strided(torch.Size([N*T, S]), [res.stride(0), res.size(2)+1]) \
                .copy_(torch.polygamma(1, ealpha.view(N*T, S)))
            Fim = res - torch.polygamma(1, ealpha.sum())
            Finv = torch.cat([x.squeeze(0).inverse().unsqueeze(0) for x in Fim.split(1)], 0)
            return torch.bmm(Finv, grad.view(N*T, S, 1)).view(N, T, S).squeeze()
        def ebeta_hook(grad):
            # see Finv, this is the batched version
            N, T, S = 1, 1, dim
            res = ebeta.new(N*T, S, S).fill_(0)
            res.as_strided(torch.Size([N*T, S]), [res.stride(0), res.size(2)+1]) \
                .copy_(torch.polygamma(1, ebeta.view(N*T, S)))
            Fim = res - torch.polygamma(1, ebeta.sum())
            Finv = torch.cat([x.squeeze(0).inverse().unsqueeze(0) for x in Fim.split(1)], 0)
            return torch.bmm(Finv, grad.view(N*T, S, 1)).view(N, T, S).squeeze()

        ealpha.register_hook(ealpha_hook)
        ebeta.register_hook(ebeta_hook)

        kl = kl_div(Dir(ealpha), Dir(ebeta))
        print("KL div: {}".format(kl.item()))
        kl.backward()

        print("ealpha: {}".format(alpha.data.clamp(-2, 5).exp()))
        #print("alpha.grad: {}".format(alpha.grad))
        print("ebeta: {}".format(beta.data.clamp(-2, 5).exp()))
        #print("beta.grad: {}".format(beta.grad))
        
        alpha.data = alpha.data - lr * alpha.grad.data
        beta.data = beta.data - lr * beta.grad.data
        alpha.grad.data.zero_()
        beta.grad.data.zero_()

        """
        lol = Finv(alpha.data) @ alpha.grad.data
        galpha = dir_natural_gradient(alpha.grad.data, alpha.view(1, 1, -1))
        gbeta = dir_natural_gradient(beta.grad.data, beta.view(1, 1, -1))
        alpha.data = alpha.data - lr * galpha.data.squeeze()
        beta.data = beta.data - lr * gbeta.data.squeeze()
        alpha.grad.data.zero_()
        beta.grad.data.zero_()
        #import pdb; pdb.set_trace()
        """
gd(alpha, beta)
nat(alpha, beta)
