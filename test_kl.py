import torch

from torch.autograd import gradcheck

from torch.distributions import Dirichlet as Dir
from torch.distributions.kl import kl_divergence as kl_div

x = torch.FloatTensor([1,1,1])
y = torch.FloatTensor([2,1,1])

def test_kl(N):
    success = True
    for i in range(N):
        x = 10 * torch.randn(15).exp()
        y = 10 * torch.randn(15).exp()
        result = gradcheck(lambda x,y: kl_div(Dir(x), Dir(y)), (x, y))
        success &= result
    return success


print(test_kl(20))
