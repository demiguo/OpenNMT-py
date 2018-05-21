from __future__ import division

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable, Function
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import onmt
from onmt.Utils import aeq, sequence_mask, logsumexp
from onmt.Models import RNNEncoder, InputFeedRNNDecoder, NMTModel

from copy import deepcopy

class MuSigma(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, mu, sigma):
        ctx.save_for_backward(mu, sigma)
        return mu, sigma

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, mu_grad_output, sigma_grad_output):
        mu, sigma = ctx.saved_variables
        #print ('----')
        #print (mu.max())
        #print (mu.min())
        #print (mu.mean())
        #print (sigma.max())
        #print (sigma.min())
        #print (sigma.mean())
        mu_grad_input = sigma_grad_input = None
        #print ('help')

        if ctx.needs_input_grad[0]:
            #mu_grad_input = mu_grad_output * sigma
            mu_grad_input = mu_grad_output
        if ctx.needs_input_grad[1]:
            #sigma_grad_input = 2*sigma*sigma*sigma_grad_output
            sigma_grad_input = sigma_grad_output
        return mu_grad_input, sigma_grad_input
musigma = MuSigma.apply

class InferenceNetwork(nn.Module):
    def __init__(self, inference_network_type, src_embeddings, tgt_embeddings,
                 rnn_type, src_layers, tgt_layers, rnn_size, dropout,
                 dist_type="none"):
        super(InferenceNetwork, self).__init__()
        self.dist_type = dist_type
        self.inference_network_type = inference_network_type
        if dist_type == "none":
            self.mask_val = float("-inf")
        else:
            self.mask_val = 1e-2

        if inference_network_type == 'embedding_only':
            #self.src_encoder = src_embeddings
            self.tgt_encoder = tgt_embeddings
        elif inference_network_type == 'brnn':
            #self.src_encoder = RNNEncoder(rnn_type, True, src_layers, rnn_size,
            #                              dropout, src_embeddings, False) 
            self.tgt_encoder = RNNEncoder(rnn_type, True, tgt_layers, rnn_size,
                                          dropout, tgt_embeddings, False) 
        elif inference_network_type == 'rnn':
            #self.src_encoder = RNNEncoder(rnn_type, False, src_layers, rnn_size,
            #                              dropout, src_embeddings, False) 
            self.tgt_encoder = RNNEncoder(rnn_type, False, tgt_layers, rnn_size,
                                          dropout, tgt_embeddings, False) 

        self.W = torch.nn.Linear(rnn_size, rnn_size, bias=False)
        self.rnn_size = rnn_size

        # to parametrize log normal distribution
        if self.dist_type == "normal":
            # TODO(demi): make 100 configurable
            self.linear_1 = nn.Linear(rnn_size + rnn_size, 500)
            self.linear_2 = nn.Linear(500, 500)
            self.mean_out = nn.Linear(500, 1)
            self.std_out  = nn.Linear(500, 1)
            self.softplus = nn.Softplus()
            self.bn_mu = nn.BatchNorm1d(1, affine=True)
            self.bn_std = nn.BatchNorm1d(1, affine=True)
            self.tanh = torch.nn.Tanh()
            self.scale_out = nn.Linear(500,1)
            self.alpha_out = nn.Linear(500,1)

    def get_normal_scores(self, h_s, h_t, src_lengths, scores):
        """ h_s: [batch x src_length x rnn_size]
            h_t: [batch x tgt_length x rnn_size]
        """
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        batch_size, tgt_, src_ = scores.size()
        aeq(src_batch, batch_size)
        aeq(src_len, src_)
        aeq(tgt_len, tgt_)
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)
        aeq(self.rnn_size, src_dim)
        
        ##import pdb; pdb.set_trace()
        #h_t_expand = h_t.unsqueeze(2).expand(-1, -1, src_len, -1)
        #h_s_expand = h_s.unsqueeze(1).expand(-1, tgt_len, -1, -1)
        ## [batch, tgt_len, src_len, src_dim]
        #h_expand = torch.cat((h_t_expand, h_s_expand), dim=3)
        #h_fold = h_expand.contiguous().view(-1, src_dim + tgt_dim)
        #
        #h_enc = self.softplus(self.linear_1(h_fold))
        #h_enc = self.softplus(self.linear_2(h_enc))
        #
        ##h_mean = self.bn_mu(self.mean_out(h_enc))
        ###h_mean = F.dropout(h_mean, p=0.1, training=self.training)
        ###h_mean = self.mean_out(h_enc)
        ###h_std = self.softplus(0.1*self.bn_std(self.std_out(h_enc)))
        ##h_std = self.softplus(self.bn_std(self.std_out(h_enc)))
        ###h_std = torch.exp(self.bn_std(self.std_out(h_enc)))
        ###h_std = self.softplus(self.std_out(h_enc))
        ##
        ##h_mean = h_mean.view(tgt_batch, tgt_len, src_len)
        ##h_std = h_std.view(tgt_batch, tgt_len, src_len)
        ###h_mean, h_std = musigma(h_mean, h_std)
        ##return [h_mean, h_std]
        mask = sequence_mask(src_lengths).view(src_batch,1,src_len).expand(-1,tgt_len,src_len).contiguous().view(src_batch*tgt_len,src_len) #batch_size, src_len
        src_lengths_tgt = src_lengths.view(src_batch, 1).float().expand(-1, tgt_len).contiguous().view(src_batch*tgt_len, 1)
        src_lengths = src_lengths.view(-1, 1, 1).float().expand(-1, tgt_len, src_len).contiguous().view(-1, src_len)
        #h_alpha_log = self.softplus(self.scale_out(h_enc)) * self.tanh(self.alpha_out(h_enc))
        h_alpha_log = scores
        #h_alpha_log = h_alpha_log.view(-1, src_len)
        #h_alpha_log.data.masked_fill_(1-mask, 0)
        #h_alpha_log_mean = h_alpha_log.sum(dim=-1, keepdim=True) / src_lengths_tgt
        #h_mean = h_alpha_log - h_alpha_log_mean
        #k1 = 1 - 2. / src_lengths
        #k2 = 1. / (src_lengths * src_lengths)
        #h_alpha = h_alpha_log.exp()
        #h_alpha_inv = 1. / h_alpha
        #h_alpha_inv.data.masked_fill_(1-mask, 0)
        #h_std = k1 * h_alpha_inv + k2 * torch.sum(h_alpha_inv, dim=-1, keepdim=True)
        #h_mean = h_mean.view(tgt_batch, tgt_len, src_len)
        #h_std = h_std.view(tgt_batch, tgt_len, src_len)
        #h_alpha = h_alpha.view(tgt_batch, tgt_len, src_len)
        return h_alpha_log.view(tgt_batch, tgt_len, src_len)

    def forward(self, src, tgt_prev, tgt, src_lengths=None, memory_bank=None):
        #src_final, src_memory_bank = self.src_encoder(src, src_lengths)
        #src_length, batch_size, rnn_size = src_memory_bank.size()
        src_memory_bank = memory_bank.transpose(0,1).transpose(1,2)
        if self.inference_network_type == 'embedding_only':
            tgt_memory_bank = self.tgt_encoder(tgt)
            tgt_memory_bank_prev = self.tgt_encoder(tgt_prev)
        else:
            tgt_final, tgt_memory_bank = self.tgt_encoder(tgt)
            #tgt_final_prev, tgt_memory_bank_prev = self.tgt_encoder(tgt_prev)
        #src_memory_bank = src_memory_bank.transpose(0,1) # batch_size, src_length, rnn_size
        #src_memory_bank = src_memory_bank.contiguous().view(-1, rnn_size) # batch_size*src_length, rnn_size
        #src_memory_bank = self.W(src_memory_bank) \
        #                      .view(batch_size, src_length, rnn_size)
        #src_memory_bank = src_memory_bank.transpose(1,2) # batch_size, rnn_size, src_length
        tgt_memory_bank = tgt_memory_bank.transpose(0,1) # batch_size, tgt_length, rnn_size
        #tgt_memory_bank_prev = tgt_memory_bank_prev.transpose(0,1) # batch_size, tgt_length, rnn_size

        if self.dist_type == "dirichlet":
            scores = torch.bmm(tgt_memory_bank, src_memory_bank)
            #print("max: {}, min: {}".format(scores.max(), scores.min()))
            # affine
            scores = scores - scores.min(-1)[0].unsqueeze(-1) + 1e-2
            # exp
            #scores = scores.clamp(-1, 1).exp()
            #scores = scores.clamp(min=1e-2)
            scores = [scores]
        elif self.dist_type == "normal":
            # log normal
            src_memory_bank = src_memory_bank.transpose(1, 2)
            batch_size, src_length, rnn_size = src_memory_bank.size()
            src_memory_bank_h = src_memory_bank.contiguous().view(-1, tgt_memory_bank.size(2))
            src_memory_bank_h = self.W(src_memory_bank_h).view(batch_size, src_length, rnn_size).transpose(1,2)
            #scores = torch.bmm(tgt_memory_bank-tgt_memory_bank_prev, src_memory_bank_h)
            scores = torch.bmm(tgt_memory_bank, src_memory_bank_h)
            print ('max: %f, min: %f'%(scores.max(), scores.min()))
            #assert src_memory_bank.size() == (batch_size, src_length, rnn_size)
            scores = self.get_normal_scores(src_memory_bank, tgt_memory_bank, src_lengths, scores)
        elif self.dist_type == "none":
            scores = [torch.bmm(tgt_memory_bank, src_memory_bank)]
        else:
            raise Exception("Unsupported dist_type")

        # length
        if src_lengths is not None:
            mask = sequence_mask(src_lengths)
            mask = mask.unsqueeze(1)
            if self.dist_type == 'normal':
                scores.data.masked_fill_(1-mask, -999)
            else:
                assert (False)
        return scores


class ViInputFeedRNNDecoder(InputFeedRNNDecoder):
    """
    Input feeding based decoder. See :obj:`RNNDecoderBase` for options.

    Based around the input feeding approach from
    "Effective Approaches to Attention-based Neural Machine Translation"
    :cite:`Luong2015`


    .. mermaid::

       graph BT
          A[Input n-1]
          AB[Input n]
          subgraph RNN
            E[Pos n-1]
            F[Pos n]
            E --> F
          end
          G[Encoder]
          H[Memory_Bank n-1]
          A --> E
          AB --> F
          E --> H
          G --> H
    """

    def _run_forward_pass(self, tgt, memory_bank, state, memory_lengths=None,
                          q_scores_residual=None):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        # Additional args check.
        input_feed = state.input_feed.squeeze(0)
        input_feed_batch, _ = input_feed.size()
        tgt_len, tgt_batch, _ = tgt.size()
        aeq(tgt_batch, input_feed_batch)
        # END Additional args check.

        if self.dist_type == "normal":
            p_a_scores = [[], [], [], []]
        else:
            assert (False)
        n_param = len(p_a_scores)
        # Initialize local and return variables.
        decoder_outputs = []
        attns = {"std": []}
        attns["q"] = []
        attns["q_raw_mean"] = []
        attns["q_raw_std"] = []
        attns["p_raw_mean"] = []
        attns["p_raw_std"] = []
        attns["p_mean_2"] = []
        attns["q_mean_2"] = []

        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim

        tgt_len, batch_size =  emb.size(0), emb.size(1)
        src_len = memory_bank.size(0)

        hidden = state.hidden
        #[item.fill_(0) for item in hidden]
        coverage = None

        # Input feed concatenates hidden state with
        # input at every time step.
        #q_scores_mean = q_scores[0].view(batch_size, tgt_len, -1).transpose(0,1)
        #q_scores_std = q_scores[1].view(batch_size, tgt_len, -1).transpose(0,1)
        q_scores_mean = []
        q_scores_std = []
        q_scores_alpha = []
        q_scores_sample = []
        q_scores_mean_2 = []
        p_scores_mean_2 = []
        if q_scores_residual is not None:
            q_scores_residual = q_scores_residual.transpose(0,1)
        for i, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            decoder_input = torch.cat([emb_t, input_feed], 1)

            rnn_output, hidden = self.rnn(decoder_input, hidden)
            decoder_output, p_attn, raw_scores, (q_scores_mean_i, q_scores_std_i, q_scores_alpha_i, q_scores_sample_i, q_scores_mean_2_i, p_scores_mean_2_i), decoder_output_input_feed = self.attn(
                rnn_output,
                memory_bank.transpose(0, 1),
                memory_lengths=memory_lengths,
                q_scores_residual=q_scores_residual[i] if q_scores_residual is not None else None)
            q_scores_mean.append(q_scores_mean_i)
            q_scores_std.append(q_scores_std_i)
            q_scores_alpha.append(q_scores_alpha_i)
            q_scores_sample.append(q_scores_sample_i)
            q_scores_mean_2.append(q_scores_mean_2_i)
            p_scores_mean_2.append(p_scores_mean_2_i)
            attns["q"] += [q_scores_sample_i.view(-1, src_len)]
            attns["q_raw_mean"] += [q_scores_mean_i.view(-1, src_len)]
            attns["q_raw_std"] += [q_scores_std_i.view(-1, src_len)]
            attns["p_raw_mean"] += [raw_scores[0].view(-1, src_len)]
            attns["p_raw_std"] += [raw_scores[1].view(-1, src_len)]
            attns["p_mean_2"] += [p_scores_mean_2_i.view(-1, src_len)]
            attns["q_mean_2"] += [q_scores_mean_2_i.view(-1, src_len)]

            # raw_scores: [batch x tgt_len x src_len]
            #assert raw_scores.size() == (batch_size, 1, src_len)

            #assert len(raw_scores) == n_param
            for i in range(len(raw_scores)):
                p_a_scores[i] += [raw_scores[i]]
            if self.context_gate is not None:
                # TODO: context gate should be employed
                # instead of second RNN transform.
                decoder_output = self.context_gate(
                    decoder_input, rnn_output, decoder_output
                )
            decoder_output = self.dropout(decoder_output)
            input_feed = decoder_output_input_feed

            decoder_outputs += [decoder_output]
            attns["std"] += [p_attn]

        q_scores_mean = torch.cat(q_scores_mean, dim=1)
        q_scores_std = torch.cat(q_scores_std, dim=1)
        q_scores_alpha = torch.cat(q_scores_alpha, dim=1)
        q_scores_sample = torch.cat(q_scores_sample, dim=1)
        for i in range(n_param):
            p_a_scores[i] = torch.cat(p_a_scores[i],dim=1)
        #if memory_lengths is not None:
        #    mask = sequence_mask(memory_lengths)
        #    mask = mask.unsqueeze(1)
        #    #if self.dist_type == 'normal':
        #    #    p_a_scores[0].data.masked_fill_(1-mask, -999)
        #    #    p_a_scores[1].data.masked_fill_(1-mask, 0.001)
        #    #    p_a_scores[2].data.masked_fill_(1-mask, 0.0001)
        #    #    p_a_scores[3].data.masked_fill_(1-mask, -999)
        #    #else:
        #    #    assert (False)
        #print ('p max: %f, min: %f'%(p_a_scores[3].max(), p_a_scores[3].min()))
        return hidden, decoder_outputs, attns, p_a_scores, [q_scores_mean, q_scores_std, q_scores_alpha]

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert not rnn_type == "SRU", "SRU doesn't support input feed! " \
                "Please set -input_feed 0!"
        if rnn_type == "LSTM":
            stacked_cell = onmt.modules.StackedLSTM
        else:
            stacked_cell = onmt.modules.StackedGRU
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)

    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors.
        """
        return self.embeddings.embedding_size + self.hidden_size


class ViNMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """
    def __init__(self, encoder, decoder, inference_network, multigpu=False, dist_type="normal"):
        self.multigpu = multigpu
        super(ViNMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.inference_network = inference_network
        self.dist_type = dist_type
        self._n_samples = 1

    @property
    def n_samples(self):
        return self._n_samples

    @n_samples.setter  
    def n_samples(self, n):
        self._n_samples = n
        #self.decoder.attn.use_prior = n > 1
        self.decoder.attn.use_prior = True
        print("Setting use_prior to: {}".format(self.decoder.attn.use_prior))

    def forward(self, src, tgt, lengths, dec_state=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        inftgt = tgt[1:]
        tgt = tgt[:-1]  # exclude last target from inputs
        tgt_length, batch_size, rnn_size = tgt.size()

        enc_final, memory_bank = self.encoder(src, lengths)
        enc_state = self.decoder.init_decoder_state(
            src, memory_bank, enc_final)
        if self.inference_network is not None:
            # inference network q(z|x,y)
            q_scores_residual = self.inference_network(src, tgt, inftgt, lengths, memory_bank) # batch_size, tgt_length, src_length
            #q_scores = self.inference_network(src, tgt, lengths, memory_bank) # batch_size, tgt_length, src_length
            src_length = q_scores_residual.size(2)
            #if self.dist_type != "none":
            #    for i in range(q_nparam):
            #        # batch_size * tgt_length, src_length
            #        q_scores[i] = q_scores[i].view(-1, q_scores[i].size(2))
            #    if self.dist_type == "dirichlet":
            #        m = torch.distributions.Dirichlet(q_scores[0].cpu())
            #    elif self.dist_type == "normal":
            #        pass
            #        #m = torch.distributions.normal.Normal(q_scores[0], q_scores[1])
            #    else:
            #        raise Exception("Unsupported dist_type")
            #    if self.dist_type == 'normal':
            #        pass
            #        #q_scores_sample = F.softmax(m.rsample().cuda(), dim=-1).view(batch_size, tgt_length, -1).transpose(0,1)
            #        #q_scores_sample = F.dropout(q_scores_sample, p=0.1, training=self.training)
            #    else:
            #        assert (False)
            #else:
            #    q_scores_sample = F.softmax(q_scores[0], dim=-1).transpose(0, 1)
        else:
            assert False

        if self.n_samples == 1:
            decoder_outputs, dec_state, attns, p_a_scores, q_scores = \
                self.decoder(tgt, memory_bank,
                             enc_state if dec_state is None
                             else dec_state,
                             memory_lengths=lengths,
                             q_scores_residual = q_scores_residual)
        else:
            outputs = []
            for _ in range(self.n_samples-1):
                decoder_outputs, _, attns, p_a_scores, q_scores = \
                    self.decoder(tgt, memory_bank,
                                 deepcopy(enc_state) if dec_state is None
                                 else deepcopy(dec_state),
                                 memory_lengths=lengths,
                                 q_scores_residual = q_scores_residual)
                outputs += [decoder_outputs]
            #decoder_outputs = logsumexp(soutputs, dim=0) - math.log(self.n_samples)
            #import pdb; pdb.set_trace()
            # LOL
            decoder_outputs, dec_state, attns, p_a_scores, q_scores = \
                self.decoder(tgt, memory_bank,
                             enc_state if dec_state is None
                             else dec_state,
                             memory_lengths=lengths,
                             q_scores_residual = q_scores_residual)
            outputs += [decoder_outputs]
            decoder_outputs = torch.stack(outputs, dim=0)

        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None

        if self.inference_network is not None:
            for i in range(len(q_scores)):
                q_scores[i] = q_scores[i].view(batch_size, tgt_length, src_length)
            if self.dist_type == "dirichlet":
                return decoder_outputs, attns, dec_state,\
                   (q_scores[0], p_a_scores[0])
            elif self.dist_type == "normal":
                return decoder_outputs, attns, dec_state,\
                   (q_scores[2], p_a_scores[2], q_scores[0], q_scores[1], p_a_scores[0], p_a_scores[1])
            elif self.dist_type == "none":
                # sigh, lol. this whole thing needs to be cleaned up
                return decoder_outputs, attns, dec_state, (
                    q_scores_sample.transpose(0,1), p_a_scores[0])
            else:
                raise Exception("Unsupported dist_type")
        else:
            return decoder_outputs, attns, dec_state, None
        # p_a_scores: feed in sampled a, output unormalized attention scores (batch_size, tgt_length, src_length)
