"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import onmt
import onmt.io


class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating mutiple
    loss computations


    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """
    def __init__(self, generator, tgt_vocab):
        super(LossComputeBase, self).__init__()
        self.generator = generator
        self.tgt_vocab = tgt_vocab
        self.padding_idx = tgt_vocab.stoi[onmt.io.PAD_WORD]

    def _make_shard_state(self, batch, output, range_, attns=None, dist_scores=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, output, target, q_scores=None, p_a_scores=None, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def monolithic_compute_loss(self, batch, output, attns, dist_scores=None):
        """
        Compute the forward loss for the batch.

        Args:
          batch (batch): batch of labeled examples
          output (:obj:`FloatTensor`):
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict of :obj:`FloatTensor`) :
              dictionary of attention distributions
              `[tgt_len x batch x src_len]`
        Returns:
            :obj:`onmt.Statistics`: loss statistics
        """
        range_ = (0, batch.tgt.size(0))
        shard_state = self._make_shard_state(batch, output, range_, attns, dist_scores=dist_scores)
        _, batch_stats = self._compute_loss(batch, **shard_state)

        return batch_stats

    def sharded_compute_loss(self, batch, output, attns,
                             cur_trunc, trunc_size, shard_size,
                             normalization, dist_scores=None):
        """Compute the forward loss and backpropagate.  Computation is done
        with shards and optionally truncation for memory efficiency.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(cur_trunc, cur_trunc + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          cur_trunc (int) : starting position of truncation window
          trunc_size (int) : length of truncation window
          shard_size (int) : maximum number of examples in a shard
          normalization (int) : Loss is divided by this number

        Returns:
            :obj:`onmt.Statistics`: validation loss statistics

        """
        batch_stats = onmt.Statistics()
        range_ = (cur_trunc, cur_trunc + trunc_size)
        shard_state = self._make_shard_state(batch, output, range_, attns, dist_scores=dist_scores)
        #print("sharded compute loss")
        #import pdb; pdb.set_trace()
        for shard in shards(shard_state, shard_size):
            loss, stats = self._compute_loss(batch, **shard)
            loss.div(normalization).backward(retain_graph=True)
            batch_stats.update(stats)

        return batch_stats

    def _stats(self, loss, xent, kl, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target) \
                          .masked_select(non_padding) \
                          .long().sum()
        return onmt.Statistics(
            loss.cpu().numpy(),
            xent.cpu().numpy(),
            kl.cpu().numpy(),
            non_padding.long().sum().cpu().numpy(),
            num_correct.cpu().numpy())

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def _unbottle(self, v, batch_size):
        return v.view(-1, batch_size, v.size(1))


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """
    def __init__(self, generator, tgt_vocab, normalization="sents",
                 label_smoothing=0.0, dist_type="none"):
        super(NMTLossCompute, self).__init__(generator, tgt_vocab)
        assert (label_smoothing >= 0.0 and label_smoothing <= 1.0)

        self.alpha = None

        # TODO(demi): change this, add KL loss for inference network
        self.dist_type = dist_type
        if label_smoothing > 0:
            # When label smoothing is turned on,
            # KL-divergence between q_{smoothed ground truth prob.}(w)
            # and p_{prob. computed by model}(w) is minimized.
            # If label smoothing value is set to zero, the loss
            # is equivalent to NLLLoss or CrossEntropyLoss.
            # All non-true labels are uniformly set to low-confidence.
            self.criterion = nn.KLDivLoss(size_average=False)
            one_hot = torch.randn(1, len(tgt_vocab))
            one_hot.fill_(label_smoothing / (len(tgt_vocab) - 2))
            one_hot[0][self.padding_idx] = 0
            self.register_buffer('one_hot', one_hot)
        else:
            weight = torch.ones(len(tgt_vocab))
            weight[self.padding_idx] = 0
            self.criterion = nn.NLLLoss(weight, size_average=False)
        self.confidence = 1.0 - label_smoothing

    def _make_shard_state(self, batch, output, range_, attns=None, dist_scores=None):
        if dist_scores is not None:
            # NB(demi): not exactly sure how sharding works yet, so let's separate q_scores and p_a_scores here for now
            if self.dist_type == "dirichlet":
                q_scores, p_a_scores = dist_scores
                q_scores = q_scores.transpose(0, 1)
                p_a_scores = p_a_scores.transpose(0, 1)
            elif self.dist_type == "normal":
                assert self.dist_type == "normal", "Dist Type not found in make shard state"
                q_scores_0, q_scores_1, p_a_scores_0, p_a_scores_1 = dist_scores
                # TODO(demi): understand why we want to transpose
                q_scores_0 = q_scores_0.transpose(0, 1)
                p_a_scores_0 = p_a_scores_0.transpose(0, 1)
                q_scores_1 = q_scores_1.transpose(0, 1)
                p_a_scores_1 = p_a_scores_1.transpose(0, 1)
            elif self.dist_type == "none":
                q_scores, p_a_scores = dist_scores
                q_scores = q_scores.transpose(0, 1)
                p_a_scores = p_a_scores.transpose(0, 1)
            else:
                raise Exception("Unsupported dist_type")
 
        else:
            q_scores, p_a_scores = None, None
        if self.dist_type == "dirichlet":
            return {
                "output": output,
                "target": batch.tgt[range_[0] + 1: range_[1]],
                "q_scores_0": q_scores,
                "p_a_scores_0": p_a_scores,
                "q_scores_1": None,
                "p_a_scores_1": None
            }
        elif self.dist_type == "normal":
            return {
                "output": output,
                "target": batch.tgt[range_[0] + 1: range_[1]],
                "q_scores_0": q_scores_0,
                "p_a_scores_0": p_a_scores_0,
                "q_scores_1": q_scores_1,
                "p_a_scores_1": p_a_scores_1
            }
        elif self.dist_type == "none":
            return {
                "output": output,
                "target": batch.tgt[range_[0] + 1: range_[1]],
                "q_scores_0": q_scores,
                "p_a_scores_0": p_a_scores,
                "q_scores_1": None,
                "p_a_scores_1": None
            }
        else:
            raise Exception("Unsupported dist_type")


    def _compute_loss(self, batch, output, target,
                      q_scores_0=None, p_a_scores_0=None, q_scores_1=None, p_a_scores_1=None):
        # TODO(demi): understand how sharding work and make sure "additional loss" works
        scores = self.generator(self._bottle(output))

        gtruth = target.view(-1)
        if self.confidence < 1:
            tdata = gtruth.data
            mask = torch.nonzero(tdata.eq(self.padding_idx)).squeeze()
            log_likelihood = torch.gather(scores.data, 1, tdata.unsqueeze(1))
            tmp_ = self.one_hot.repeat(gtruth.size(0), 1)
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)
            if mask.dim() > 0:
                log_likelihood.index_fill_(0, mask, 0)
                tmp_.index_fill_(0, mask, 0)
            gtruth = Variable(tmp_, requires_grad=False)
        xent = self.criterion(scores, gtruth)
        
        if q_scores_0 is None or p_a_scores_0 is None:
            loss = xent
            kl = xent.new([0])
        elif self.dist_type == "dirichlet":
            q_scores_0 = q_scores_0.contiguous().view(-1, q_scores_0.size(2))
            p_a_scores_0 = p_a_scores_0.contiguous().view(-1, p_a_scores_0.size(2))
            q_scores_0 = q_scores_0[gtruth.ne(self.padding_idx)]
            p_a_scores_0 = p_a_scores_0[gtruth.ne(self.padding_idx)]

            q_dist = torch.distributions.Dirichlet(q_scores_0.detach())
            p_a_dist = torch.distributions.Dirichlet(p_a_scores_0.detach())
            kl = torch.distributions.kl.kl_divergence(q_dist, p_a_dist).sum()
            assert xent.size() == kl.size(), "xent.size():{}\nkl.size():{}\n".format(xent.size(), kl.size())
            #loss += kl
            loss = xent # + kl
        elif self.dist_type == "normal":
            q_scores_0 = q_scores_0.contiguous().view(-1, q_scores_0.size(2))
            p_a_scores_0 = p_a_scores_0.contiguous().view(-1, p_a_scores_0.size(2))
            q_scores_0 = q_scores_0[gtruth.ne(self.padding_idx)]
            p_a_scores_0 = p_a_scores_0[gtruth.ne(self.padding_idx)]

            q_scores_1 = q_scores_1.contiguous().view(-1, q_scores_1.size(2))
            p_a_scores_1 = p_a_scores_1.contiguous().view(-1, p_a_scores_1.size(2))
            # 64, 15, 12
            q_scores_1 = q_scores_1[gtruth.ne(self.padding_idx)]
            p_a_scores_1 = p_a_scores_1[gtruth.ne(self.padding_idx)]

            q_dist = torch.distributions.normal.Normal(q_scores_0, q_scores_1)
            p_a_dist = torch.distributions.normal.Normal(p_a_scores_0, p_a_scores_1)

            kl = torch.distributions.kl.kl_divergence(q_dist, p_a_dist).sum()
            assert xent.size() == kl.size(), "xent.size():{}\nkl.size():{}\n".format(xent.size(), kl.size())
        elif self.dist_type == "none":
            # Minimize KL from sample, lol.
            q_scores_0 = q_scores_0.contiguous().view(-1, q_scores_0.size(2))
            p_a_scores_0 = p_a_scores_0.contiguous().view(-1, p_a_scores_0.size(2))
            q_scores_0 = q_scores_0[gtruth.ne(self.padding_idx)]
            p_a_scores_0 = p_a_scores_0[gtruth.ne(self.padding_idx)]

            q_dist = torch.distributions.Categorical(q_scores_0)
            p_a_dist = torch.distributions.Categorical(p_a_scores_0)
            kl = torch.distributions.kl.kl_divergence(q_dist, p_a_dist).sum()
        else:
            raise Exception("Unsupported dist_type")

        if self.alpha is not None:
            loss = xent + self.alpha * kl
        else:
            loss = xent + kl

        #loss = xent + kl*0.5
        if self.confidence < 1:
            # Default: report smoothed ppl.
            # loss_data = -log_likelihood.sum(0)
            loss_data = loss.data.clone()
        else:
            loss_data = loss.data.clone()

        stats = self._stats(loss_data, xent.data.clone(), kl.data.clone(), scores.data, target.view(-1).data)

        return loss, stats


def filter_shard_state(state, requires_grad=True):
    for k, v in state.items():
        if v is not None:
            if isinstance(v, Variable) and v.requires_grad:
                v = Variable(v.data, requires_grad=requires_grad)
            yield k, v


def shards(state, shard_size, eval=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval:
        yield filter_shard_state(state, False)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, torch.split(v, shard_size))
                             for k, v in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = ((state[k], v.grad.data) for k, v in non_none.items()
                     if isinstance(v, Variable) and v.grad is not None)
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)
