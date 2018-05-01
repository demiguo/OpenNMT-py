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
    args.add_argument("--devid", type=int, default=-1)
    args.add_argument("--worstn", type=int, default=10)
    args.add_argument("--port", type=int, default=8097)
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
#model.cuda(args.devid) # lol

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
    q_attns = []
    wordnlls = []
    for i, example in tqdm(enumerate(valid)):
        if i > 5:
            break
        x = srcfield.process([example.src], device=devid, train=False)
        y = tgtfield.process([example.tgt], device=devid, train=False)

        if True:
            #output, attn_dict, decoderstate, (q_scores, p_a_scores) = model(x[0].view(-1, 1, 1), y.view(-1, 1, 1), x[1])
            output, attn_dict, decoderstate, (q_scores, p_a_scores) = model(
                x[0].view(-1, 1, 1).cpu(), y.view(-1, 1, 1).cpu(), x[1].cpu())
            attn = attn_dict["std"]
            q_attn = attn_dict["q"]
            import pdb; pdb.set_trace()

            lsm = model.generator(output.squeeze(1))

            bloss = F.nll_loss(lsm, y.view(-1)[1:], reduce=False)

            nlls[i] = bloss.mean().data[0]
            attns.append(attn)
            q_attns.append(q_attn)
            wordnlls.append(bloss)

    """
    torch.save(nlls, nllpath)
    torch.save(attns, attnpath)
    torch.save(wordnlls, wordpath)
    """


def visualize_attn():
    import visdom
    vis = visdom.Visdom(port=args.port)
    #for i in np.random.permutation(len(valid))[:5]:
    #for i in [3105, 2776, 2424, 2357, 1832]:
    for i in [0, 1, 2, 3]:
        example = valid[i]
        attn = attns[i]

        # unused
        nll = nlls[i]
        wordnll = wordnlls[i]

        rownames = list(example.tgt) + ["<eos>"]
        rownames = ["[{}] {} ({:.2f})".format(i, name, wordnll[i].data[0]) for i, name in enumerate(rownames)]
        columnnames = list(example.src)
        columnnames = ["[{}] {}".format(i, name) for i, name in enumerate(columnnames)]
        title = "Gen Model {} Example {}".format(args.checkpoint_path.split("/")[-2], i)
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

        attn = q_attns[i]

        # unused
        nll = nlls[i]
        wordnll = wordnlls[i]

        rownames = list(example.tgt) + ["<eos>"]
        rownames = ["[{}] {} ({:.2f})".format(i, name, wordnll[i].data[0]) for i, name in enumerate(rownames)]
        columnnames = list(example.src)
        columnnames = ["[{}] {}".format(i, name) for i, name in enumerate(columnnames)]
        title = "Inf Net {} Example {}".format(args.checkpoint_path.split("/")[-2], i)
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


visualize_attn()
idx = 2357
example = valid[idx]
srcfield = fields["src"]
tgtfield = fields["tgt"]
x = srcfield.numericalize(srcfield.pad([example.src]), device=devid, train=False)
y = tgtfield.numericalize(tgtfield.pad([example.tgt]), device=devid, train=False)
#import pdb; pdb.set_trace()
