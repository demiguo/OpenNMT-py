import torch
import torch.nn.functional as F

def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def use_gpu(opt):
    return (hasattr(opt, 'gpuid') and len(opt.gpuid) > 0) or \
        (hasattr(opt, 'gpu') and opt.gpu > -1)


def sample_attn(scores, dist_type):
    """ Fix later, I guess """
    batch_size, tgt_length, src_length = scores[0].size()
    # scores : N x T x S
    nparam = len(scores)
    if dist_type != "none":
        # batch_size * tgt_length, src_length
        scores = [x.view(-1, x.size(-1)) for x in scores]
        if dist_type == "dirichlet":
            m = torch.distributions.Dirichlet(scores[0].cpu())
        elif dist_type == "normal":
            m = torch.distributions.normal.Normal(scores[0], scores[1])
        else:
            raise Exception("Unsupported dist_type")
        if dist_type == 'normal':
            sample = F.softmax(m.rsample().cuda(), dim=-1).view(batch_size, tgt_length, -1).transpose(0,1)
            #sample = F.dropout(sample, p=0.1, training=training)
        else:
            sample = m.rsample().cuda().view(batch_size, tgt_length, -1).transpose(0,1)
    else:
        sample = F.softmax(scores[0], dim=-1).transpose(0, 1)
    # output is T x N x S, LOL
    return sample
