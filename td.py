import torch

from torch.autograd import backward, grad

from torch.distributions import Dirichlet as Dir
from torch.distributions.kl import kl_divergence as kl_div

import torch.nn.functional as F

steps = 10
lr = 1

dim = 20
alpha = torch.randn(dim, requires_grad=True)
beta = torch.rand(dim, requires_grad=True) * 4 - 2

true = Dir(4 * torch.randn(dim))
data = true.rsample(torch.Size([256]))

print(alpha.data.view(4, 5))
print(beta.data.view(4, 5))

UPDATE_BETA = False

def get_kl(alpha, beta):
    return kl_div(Dir(alpha), Dir(beta))

# GD
def gd(alpha, beta):
    lr = 1
    print("GRADIENT DESCENT")
    alpha, beta = alpha.detach(), beta.detach()
    alpha.requires_grad = True
    beta.requires_grad = True
    for i in range(steps):
        print("Step {}".format(i))
        # exponentiate?
        kl = get_kl(F.softplus(alpha), F.softplus(beta))
        print("KL div: {}".format(kl.item()))
        kl.backward()
        #print("ealpha: {}".format(alpha.data.clamp(-2, 5).exp().view(4,5)))
        #print("alpha.grad: {}".format(alpha.grad))
        #print("ebeta: {}".format(beta.data.clamp(-2, 5).exp().view(4,5)))
        #print("beta.grad: {}".format(beta.grad))

        alpha.data = alpha.data - lr * alpha.grad.data
        if UPDATE_BETA:
            beta.data = beta.data - lr * beta.grad.data
        alpha.grad.data.zero_()
        beta.grad.data.zero_()
    print("alpha: {}".format(alpha.data.view(4, 5)))
    print("beta: {}".format(beta.data.view(4, 5)))

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
        dalpha = F.softplus(alpha)
        dbeta = F.softplus(beta)

        def dalpha_hook(grad):
            # see Finv, this is the batched version
            N, T, S = 1, 1, dim
            res = dalpha.new(N*T, S, S).fill_(0)
            res.as_strided(torch.Size([N*T, S]), [res.stride(0), res.size(2)+1]) \
                .copy_(torch.polygamma(1, dalpha.view(N*T, S)))
            Fim = res - torch.polygamma(1, dalpha.sum())
            Finv = torch.cat([x.squeeze(0).inverse().unsqueeze(0) for x in Fim.split(1)], 0)
            return torch.bmm(Finv, grad.view(N*T, S, 1)).view(N, T, S).squeeze()
        def dbeta_hook(grad):
            # see Finv, this is the batched version
            N, T, S = 1, 1, dim
            res = dbeta.new(N*T, S, S).fill_(0)
            res.as_strided(torch.Size([N*T, S]), [res.stride(0), res.size(2)+1]) \
                .copy_(torch.polygamma(1, dbeta.view(N*T, S)))
            Fim = res - torch.polygamma(1, dbeta.sum())
            Finv = torch.cat([x.squeeze(0).inverse().unsqueeze(0) for x in Fim.split(1)], 0)
            return torch.bmm(Finv, grad.view(N*T, S, 1)).view(N, T, S).squeeze()

        dalpha.register_hook(dalpha_hook)
        if UPDATE_BETA:
            dbeta.register_hook(dbeta_hook)

        kl = kl_div(Dir(dalpha), Dir(dbeta))
        print("KL div: {}".format(kl.item()))
        kl.backward()

        #print("alpha: {}".format(alpha.data.view(4,5)))
        #print("alpha.grad: {}".format(alpha.grad))
        #print("beta: {}".format(beta.data.view(4,5)))
        #print("beta.grad: {}".format(beta.grad))

        """
        lol = Finv(alpha.data) @ alpha.grad.data
        galpha = dir_natural_gradient(alpha.grad.data, alpha.view(1, 1, -1))
        gbeta = dir_natural_gradient(beta.grad.data, beta.view(1, 1, -1))
        alpha.data = alpha.data - lr * galpha.data.squeeze()
        if UPDATE_BETA:
            beta.data = beta.data - lr * gbeta.data.squeeze()
        """

        alpha.data = alpha.data - lr * alpha.grad.data
        if UPDATE_BETA:
            beta.data = beta.data - lr * beta.grad.data

        alpha.grad.data.zero_()
        beta.grad.data.zero_()

    print("alpha: {}".format(alpha.data.view(4, 5)))
    print("beta: {}".format(beta.data.view(4, 5)))

gd(alpha, beta)
nat(alpha, beta)
