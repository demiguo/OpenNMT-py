import argparse
import os
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable as V

from typing import NamedTuple

import numpy as np

import onmt
import onmt.ModelConstructor

#from tensorboardX import SummaryWriter
import visdom

from tqdm import tqdm

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--data", type=str,
        default=None,
        required=True)
    args.add_argument(
        "--checkpoint_path", type=str,
        default=None,
        required=True)
    args.add_argument("--savepath", type=str, default=None)
    args.add_argument("--modelname", type=str, default=None, required=True)
    args.add_argument("--devid", type=int, default=0)
    args.add_argument("--worstn", type=int, default=10)
    return args.parse_args()


args = get_args()

devid = args.devid

train = torch.load(args.data + '.train.1.pt')
valid = torch.load(args.data + '.valid.1.pt')

fields = onmt.io.load_fields_from_vocab(
    torch.load(args.data + '.vocab.pt'))
# Why do we have to filter fields? Seems hacky.
fields = dict([(k, f) for (k, f) in fields.items()
    if k in train.examples[0].__dict__])
train.fields = fields
valid.fields = fields
savepath = os.path.join(args.savepath, args.modelname)
nllpath = savepath + ".nlls"
attnpath = savepath + ".attns"
wordpath = savepath + ".wordnlls"

checkpoint = torch.load(
    args.checkpoint_path,
    map_location=lambda storage, loc: storage)
model_opt = checkpoint['opt']
model = onmt.ModelConstructor.make_base_model(model_opt, fields, devid, checkpoint)
model.cuda() # lol

nlls = None
attns = None
wordnlls = None
if False and os.path.isfile(nllpath) and os.path.isfile(attnpath) and os.path.isfile(wordpath):
    nlls = torch.load(nllpath)
    attns = torch.load(attnpath)
    wordnlls = torch.load(wordpath)
else:
    srcfield = fields["src"]
    tgtfield = fields["tgt"]
    # Sentence, NLL pairs, sorted decreasing (worst sentences first)
    # And attention scores
    nlls = torch.FloatTensor(len(valid))
    attns = []
    attns_mean = []
    attns_std = []
    ps_mean = []
    ps_std = []
    wordnlls = []
    for i, example in tqdm(enumerate(valid)):
        if i > 5:
            break
        x = srcfield.process([example.src], device=devid, train=False)
        y = tgtfield.process([example.tgt], device=devid, train=False)

        if True:
            if model_opt.dist_type == 'normal':
                output, attn_dict, decoderstate, (_, _, q_scores, q_scores_std, p_a_scores, p_a_score_std) = model(x[0].view(-1, 1, 1), y.view(-1, 1, 1), x[1])
            else:
                output, attn_dict, decoderstate, (q_scores, p_a_scores) = model(x[0].view(-1, 1, 1), y.view(-1, 1, 1), x[1])
            #attn = attn_dict["std"]
            attn = attn_dict["q"]
            attn_mean = attn_dict["q_raw_mean"]
            attn_std = attn_dict["q_raw_std"]
            p_mean = attn_dict["p_raw_mean"]
            p_std = attn_dict["p_raw_std"]
            #import pdb; pdb.set_trace()

            lsm = model.generator(output.squeeze(1))

            bloss = F.nll_loss(lsm, y.view(-1)[1:], reduce=False)

            nlls[i] = bloss.mean().data[0]
            attns.append(attn)
            attns_mean.append(attn_mean)
            attns_std.append(attn_std)
            ps_mean.append(p_mean)
            ps_std.append(p_std)
            wordnlls.append(bloss)

    """
    torch.save(nlls, nllpath)
    torch.save(attns, attnpath)
    torch.save(wordnlls, wordpath)
    """


def visualize_attn():
    import visdom
    vis = visdom.Visdom()
    #for i in np.random.permutation(len(valid))[:5]:
    #for i in [3105, 2776, 2424, 2357, 1832]:
    for i in [0, 1, 2, 3]:
        example = valid[i]
        attn = attns[i]
        attn_mean = attns_mean[i]
        attn_std = attns_std[i]
        p_mean = ps_mean[i]
        p_std = ps_std[i]

        # unused
        nll = nlls[i]
        wordnll = wordnlls[i]

        rownames = list(example.tgt) + ["<eos>"]
        rownames = ["[{}] {} ({:.2f})".format(i, name, wordnll[i].data[0]) for i, name in enumerate(rownames)]
        columnnames = list(example.src)
        columnnames = ["[{}] {}".format(i, name) for i, name in enumerate(columnnames)]
        title = "Sampled Q {} Example {}".format(args.checkpoint_path, i)
        vis.heatmap(
            X=attn.data.cpu().squeeze(),
            opts=dict(
                rownames=rownames,
                columnnames=columnnames,
                colormap="Hot",
                title=title,
                width=750,
                height=750,
                marginleft=150,
                marginright=150,
                margintop=150,
                marginbottom=150
            ),
            win=title
        )
        rownames = list(example.tgt) + ["<eos>"]
        rownames = ["[{}] {} ({:.2f})".format(i, name, wordnll[i].data[0]) for i, name in enumerate(rownames)]
        columnnames = list(example.src)
        columnnames = ["[{}] {}".format(i, name) for i, name in enumerate(columnnames)]
        title = "Raw Mean {} Example {}".format(args.checkpoint_path, i)
        vis.heatmap(
            X=attn_mean.data.cpu().squeeze(),
            opts=dict(
                rownames=rownames,
                columnnames=columnnames,
                colormap="Hot",
                title=title,
                width=750,
                height=750,
                marginleft=150,
                marginright=150,
                margintop=150,
                marginbottom=150
            ),
            win=title
        )
        rownames = list(example.tgt) + ["<eos>"]
        rownames = ["[{}] {} ({:.2f})".format(i, name, wordnll[i].data[0]) for i, name in enumerate(rownames)]
        columnnames = list(example.src)
        columnnames = ["[{}] {}".format(i, name) for i, name in enumerate(columnnames)]
        title = "Raw Std {} Example {}".format(args.checkpoint_path, i)
        vis.heatmap(
            X=attn_std.data.cpu().squeeze(),
            opts=dict(
                rownames=rownames,
                columnnames=columnnames,
                colormap="Hot",
                title=title,
                width=750,
                height=750,
                marginleft=150,
                marginright=150,
                margintop=150,
                marginbottom=150
            ),
            win=title
        )
        rownames = list(example.tgt) + ["<eos>"]
        rownames = ["[{}] {} ({:.2f})".format(i, name, wordnll[i].data[0]) for i, name in enumerate(rownames)]
        columnnames = list(example.src)
        columnnames = ["[{}] {}".format(i, name) for i, name in enumerate(columnnames)]
        title = "P Raw Mean {} Example {}".format(args.checkpoint_path, i)
        vis.heatmap(
            X=p_mean.data.cpu().squeeze(),
            opts=dict(
                rownames=rownames,
                columnnames=columnnames,
                colormap="Hot",
                title=title,
                width=750,
                height=750,
                marginleft=150,
                marginright=150,
                margintop=150,
                marginbottom=150
            ),
            win=title
        )
        rownames = list(example.tgt) + ["<eos>"]
        rownames = ["[{}] {} ({:.2f})".format(i, name, wordnll[i].data[0]) for i, name in enumerate(rownames)]
        columnnames = list(example.src)
        columnnames = ["[{}] {}".format(i, name) for i, name in enumerate(columnnames)]
        title = "P Raw Std {} Example {}".format(args.checkpoint_path, i)
        vis.heatmap(
            X=p_std.data.cpu().squeeze(),
            opts=dict(
                rownames=rownames,
                columnnames=columnnames,
                colormap="Hot",
                title=title,
                width=750,
                height=750,
                marginleft=150,
                marginright=150,
                margintop=150,
                marginbottom=150
            ),
            win=title
        )


visualize_attn()
idx = 2357
example = valid[idx]
srcfield = fields["src"]
tgtfield = fields["tgt"]
x = srcfield.numericalize(srcfield.pad([example.src]), device=devid, train=False)
y = tgtfield.numericalize(tgtfield.pad([example.tgt]), device=devid, train=False)
import pdb; pdb.set_trace()
