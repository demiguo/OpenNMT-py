from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import onmt
from onmt.Utils import aeq, sequence_mask, sample_attn
from onmt.Models import RNNEncoder, InputFeedRNNDecoder, NMTModel


class InferenceNetwork(nn.Module):
    def __init__(self, inference_network_type, src_embeddings, tgt_embeddings,
                 rnn_type, src_layers, tgt_layers, rnn_size, dropout,
                 attn_type="mlp",
                 dist_type="none", norm_alpha=1.0, norm_beta=1.0,
                 normalization="bn"):
        super(InferenceNetwork, self).__init__()
        self.attn_type = attn_type
        self.dist_type = dist_type
        self.inference_network_type = inference_network_type
        self.normalization = normalization

        # trainable alpha and beta
        self.mean_norm_alpha = nn.Parameter(torch.FloatTensor([1.]))
        self.mean_norm_beta = nn.Parameter(torch.FloatTensor([0.]))
        self.std_norm_alpha = nn.Parameter(torch.FloatTensor([1.]))
        self.std_norm_beta = nn.Parameter(torch.FloatTensor([0.]))

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

        self.W = torch.nn.Linear(rnn_size, rnn_size)
        self.rnn_size = rnn_size

        # to parametrize log normal distribution
        H = rnn_size
        if self.attn_type == "general":
            self.linear_in = nn.Linear(H, H, bias=False)
            if self.dist_type == "normal":
                self.W_mu = self.linear_in
                self.W_sigma = nn.Linear(H, H, bias=False)
        elif self.attn_type == "mlpadd":
            self.linear_context = nn.Linear(H, H, bias=False) 
            self.linear_query = nn.Linear(H, H, bias=True)
            self.v = nn.Linear(H, 1, bias=False)
            if self.dist_type == "normal":
                self.v_mu = self.v
                self.v_sigma = nn.Linear(H, 1, bias=False)
        elif self.attn_type == "mlp":
            if self.dist_type == "normal":
                # TODO(demi): make 100 configurable
                self.linear_1 = nn.Linear(rnn_size + rnn_size, 500)
                self.linear_2 = nn.Linear(500, 500)
                self.mean_out = nn.Linear(500, 1)
                self.std_out  = nn.Linear(500, 1)

                self.softplus = nn.Softplus()
        elif self.attn_type == "dotmlp":
            self.linear_in = nn.Linear(H, H, bias=False)
            if self.dist_type == "normal":
                self.W_mu = self.linear_in
            pass # unfinished

        if self.normalization == "bn":
            if self.dist_type == "normal":
                self.bn_mu = nn.BatchNorm1d(1, affine=True)
                self.bn_std = nn.BatchNorm1d(1, affine=True)
        elif self.normalization == "ln":
            if self.dist_type == "normal":
                self.mean_norm_alpha = nn.Parameter(torch.Tensor([1]))
                self.std_norm_alpha = nn.Parameter(torch.Tensor([1]))
                self.mean_norm_beta = nn.Parameter(torch.Tensor([0]))
                self.std_norm_beta = nn.Parameter(torch.Tensor([0]))
        elif self.normalization == "lnsigma":
            if self.dist_type == "normal":
                self.mean_norm_beta = nn.Parameter(torch.Tensor([0]))
                self.std_norm_beta = nn.Parameter(torch.Tensor([0]))

    def get_normal_scores(self, h_s, h_t):
        """ h_s: [batch x src_length x rnn_size]
            h_t: [batch x tgt_length x rnn_size]
        """

        if self.attn_type == "mlp":
            src_batch, src_len, src_dim = h_s.size()
            tgt_batch, tgt_len, tgt_dim = h_t.size()
            aeq(src_batch, tgt_batch)
            aeq(src_dim, tgt_dim)
            aeq(self.rnn_size, src_dim)
            
            #import pdb; pdb.set_trace()
            h_t_expand = h_t.unsqueeze(2).expand(-1, -1, src_len, -1)
            h_s_expand = h_s.unsqueeze(1).expand(-1, tgt_len, -1, -1)
            # [batch, tgt_len, src_len, src_dim]
            h_expand = torch.cat((h_t_expand, h_s_expand), dim=3)
            h_fold = h_expand.contiguous().view(-1, src_dim + tgt_dim)
            
            h_enc = self.softplus(self.linear_1(h_fold))
            h_enc = self.softplus(self.linear_2(h_enc))
            
            h_mean = self.mean_out(h_enc)
            #h_mean = F.dropout(h_mean, p=0.1, training=self.training)
            #h_mean = self.mean_out(h_enc)
            #h_std = self.softplus(0.1*self.bn_std(self.std_out(h_enc)))
            h_std = self.std_out(h_enc)
            #h_std = torch.exp(self.bn_std(self.std_out(h_enc)))
            #h_std = self.softplus(self.std_out(h_enc))

            if self.normalization == "bn":
                # BN
                h_mean = self.bn_mu(h_mean)
                h_std = self.softplus(self.bn_std(h_std))
                
                h_mean = h_mean.view(tgt_batch, tgt_len, src_len)
                h_std = h_std.view(tgt_batch, tgt_len, src_len)
            elif self.normalization == "ln":
                # LN
                h_mean = h_mean.view(tgt_batch, tgt_len, src_len)
                h_std = h_std.view(tgt_batch, tgt_len, src_len)

                h_mean_row_mean = torch.mean(h_mean, dim=2, keepdim=True).expand(tgt_batch, tgt_len, src_len)
                h_mean_row_std = torch.std(h_mean, dim=2, keepdim=True).expand(tgt_batch, tgt_len, src_len)
                
                h_std_row_mean = torch.mean(h_std, dim=2, keepdim=True).expand(tgt_batch, tgt_len, src_len)
                h_std_row_std = torch.std(h_std, dim=2, keepdim=True).expand(tgt_batch, tgt_len, src_len)

                h_mean = self.mean_norm_alpha * (h_mean - h_mean_row_mean) / h_mean_row_std + self.mean_norm_beta
                h_std = self.std_norm_alpha * (h_std - h_std_row_mean) / h_std_row_std + self.std_norm_beta
                h_std = self.softplus(h_std)
            elif self.normalization == "lnsigma":
                # LN on sigma only
                h_mean = h_mean.view(tgt_batch, tgt_len, src_len)
                h_std = h_std.view(tgt_batch, tgt_len, src_len)

                h_std_row_mean = torch.mean(h_std, dim=2, keepdim=True).expand(tgt_batch, tgt_len, src_len)
                h_std_row_std = torch.std(h_std, dim=2, keepdim=True).expand(tgt_batch, tgt_len, src_len)

                h_std = self.std_norm_alpha * (h_std - h_std_row_mean) / h_std_row_std + self.std_norm_beta
                h_std = self.softplus(h_std)
            elif self.normalization == "clampsigma":
                h_mean = h_mean.view(tgt_batch, tgt_len, src_len)
                h_std = h_std.view(tgt_batch, tgt_len, src_len)
                h_std = self.softplus(h_std).clamp(max=1)

            return [h_mean, h_std]

        elif self.attn_type == "mlpadd":
            # unimplemented switching on norm
            H = self.rnn_size
            Ns, S, Hs = h_s.size()
            Nt, T, Ht = h_t.size()
            aeq(Ns, Nt)
            aeq(Hs, Ht)
            aeq(Hs, H)

            wq = self.linear_query(h_t.contiguous().view(-1, H))
            wq = wq.view(Nt, T, 1, H).expand(Nt, T, S, H)

            uh = self.linear_context(h_s.contiguous().view(-1, H))
            uh = uh.view(Ns, 1, S, H).expand(Ns, T, S, H)

            wquh = F.softplus(wq + uh)

            h_mu = self.bn_mu(self.v_mu(wquh.view(-1, H))).view(Nt, T, S)
            #h_mu_un = self.v_mu(wquh.view(-1, H)).view(Nt, T, S)
            #h_mu_mean = h_mu_un.mean(2, keepdim=True)
            #h_mu = h_mu_un - h_mu_mean
            h_sigma = F.softplus(self.bn_std(self.v_sigma(wquh.view(-1, H)))).view(Nt, T, S)

            return [h_mu, h_sigma]
        elif self.attn_type == "general":
            # unimplemented switching on norm
            H = self.rnn_size
            Ns, S, Hs = h_s.size()
            Nt, T, Ht = h_t.size()
            aeq(Ns, Nt)
            aeq(Hs, Ht)
            aeq(Hs, H)

            h_t = h_t.contiguous()
            h_t_mu = self.W_mu(h_t.view(Nt * T, Ht)).view(Nt, T, H)
            h_t_sigma = self.W_sigma(h_t.view(Nt * T, Ht)).view(Nt, T, H)

            h_s_ = h_s.transpose(1, 2)

            h_mu = self.bn_mu(torch.bmm(h_t_mu, h_s_).view(-1, 1)).view(Nt, T, S)
            #h_mu_un = torch.bmm(h_t_mu, h_s_).view(Nt, T, S)
            #h_mu_mean = h_mu_un.mean(2, keepdim=True)
            #h_mu = h_mu_un - h_mu_mean
            h_sigma = F.softplus(self.bn_std(torch.bmm(h_t_sigma, h_s_).view(-1, 1))).view(Nt, T, S)
            return [h_mu, h_sigma]

    def forward(self, src, tgt, src_lengths=None, memory_bank=None):
        #src_final, src_memory_bank = self.src_encoder(src, src_lengths)
        #src_length, batch_size, rnn_size = src_memory_bank.size()
        src_memory_bank = memory_bank.transpose(0,1).transpose(1,2)
        if self.inference_network_type == 'embedding_only':
            tgt_memory_bank = self.tgt_encoder(tgt)
        else:
            tgt_final, tgt_memory_bank = self.tgt_encoder(tgt)
        #src_memory_bank = src_memory_bank.transpose(0,1) # batch_size, src_length, rnn_size
        #src_memory_bank = src_memory_bank.contiguous().view(-1, rnn_size) # batch_size*src_length, rnn_size
        #src_memory_bank = self.W(src_memory_bank) \
        #                      .view(batch_size, src_length, rnn_size)
        #src_memory_bank = src_memory_bank.transpose(1,2) # batch_size, rnn_size, src_length
        tgt_memory_bank = tgt_memory_bank.transpose(0,1) # batch_size, tgt_length, rnn_size

        if self.dist_type == "dirichlet":
            # probably broken
            scores = torch.bmm(tgt_memory_bank, src_memory_bank)
            scores = [scores]
        elif self.dist_type == "normal":
            # log normal
            src_memory_bank = src_memory_bank.transpose(1, 2)
            #assert src_memory_bank.size() == (batch_size, src_length, rnn_size)
            scores = self.get_normal_scores(src_memory_bank, tgt_memory_bank)
        elif self.dist_type == "none":
            scores = [torch.bmm(tgt_memory_bank, src_memory_bank)]
        else:
            raise Exception("Unsupported dist_type")

        nparam = len(scores)
        # length
        if src_lengths is not None:
            mask = sequence_mask(src_lengths)
            mask = mask.unsqueeze(1)
            if self.dist_type == 'normal':
                scores[0].data.masked_fill_(1-mask, -999)
                scores[1].data.masked_fill_(1-mask, 0.001)
            else:
                for i in range(nparam):
                    scores[i].data.masked_fill_(1-mask, self.mask_val)
        return scores


class ViInputFeedRNNDecoder(InputFeedRNNDecoder):
    def _run_forward_pass(self, tgt, memory_bank, state, memory_lengths=None,
                          q_scores_sample=None, q_scores=None):
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

        if self.dist_type == "dirichlet":
            p_a_scores = [[]]
        elif self.dist_type == "normal":
            p_a_scores = [[], []]
        else:
            p_a_scores = [[]]
        n_param = len(p_a_scores)
        # Initialize local and return variables.
        decoder_outputs = []
        attns = {"std": []}
        if q_scores_sample is not None:
            attns["q"] = []
        if q_scores is not None:
            attns["q_raw_mean"] = []
            attns["q_raw_std"] = []
            attns["p_raw_mean"] = []
            attns["p_raw_std"] = []
        if self._copy:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim

        tgt_len, batch_size =  emb.size(0), emb.size(1)
        src_len = memory_bank.size(0)

        hidden = state.hidden
        #[item.fill_(0) for item in hidden]
        coverage = state.coverage.squeeze(0) \
            if state.coverage is not None else None

        # Input feed concatenates hidden state with
        # input at every time step.
        if q_scores is not None:
            q_scores_mean = q_scores[0].view(batch_size, tgt_len, -1).transpose(0,1)
            q_scores_std = q_scores[1].view(batch_size, tgt_len, -1).transpose(0,1)
        for i, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            decoder_input = torch.cat([emb_t, input_feed], 1)

            rnn_output, hidden = self.rnn(decoder_input, hidden)
            if q_scores_sample is not None:
                q_sample = q_scores_sample[i]
            else:
                q_sample = None
            decoder_output, p_attn, raw_scores = self.attn(
                rnn_output,
                memory_bank.transpose(0, 1),
                memory_lengths=memory_lengths,
                q_scores_sample=q_sample)
            if q_sample is not None:
                attns["q"] += [q_sample]
                attns["q_raw_mean"] += [q_scores_mean[i]]
                attns["q_raw_std"] += [q_scores_std[i]]
                attns["p_raw_mean"] += [raw_scores[0].view(-1, src_len)]
                attns["p_raw_std"] += [raw_scores[1].view(-1, src_len)]

            # raw_scores: [batch x tgt_len x src_len]
            #assert raw_scores.size() == (batch_size, 1, src_len)

            assert len(raw_scores) == n_param
            for i in range(n_param):
                p_a_scores[i] += [raw_scores[i]]
            if self.context_gate is not None:
                # TODO: context gate should be employed
                # instead of second RNN transform.
                decoder_output = self.context_gate(
                    decoder_input, rnn_output, decoder_output
                )
            decoder_output = self.dropout(decoder_output)
            input_feed = decoder_output

            decoder_outputs += [decoder_output]
            attns["std"] += [p_attn]

            # Update the coverage attention.
            if self._coverage:
                coverage = coverage + p_attn \
                    if coverage is not None else p_attn
                attns["coverage"] += [coverage]

            # Run the forward pass of the copy attention layer.
            if self._copy and not self._reuse_copy_attn:
                _, copy_attn = self.copy_attn(decoder_output,
                                              memory_bank.transpose(0, 1))
                attns["copy"] += [copy_attn]
            elif self._copy:
                attns["copy"] = attns["std"]

        for i in range(n_param):
            p_a_scores[i] = torch.cat(p_a_scores[i],dim=1)
        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths)
            mask = mask.unsqueeze(1)
            if self.dist_type == 'normal':
                p_a_scores[0].data.masked_fill_(1-mask, -999)
                p_a_scores[1].data.masked_fill_(1-mask, 0.001)
            else:
                for i in range(n_param):
                    p_a_scores[i].data.masked_fill_(1-mask, 1e-2)
        return hidden, decoder_outputs, attns, p_a_scores

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
    def __init__(self, encoder, decoder, inference_network, multigpu=False, dist_type="normal", use_prior=False):
        self.multigpu = multigpu
        super(ViNMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.inference_network = inference_network
        self.dist_type = dist_type
        self.use_prior = use_prior
        # use this during decoding
        self.no_q = False

        self.approximate_ppl = True
        self.approximate_ppl_nsample = 5
        
        # TODO(demi): add "approximate ppl & approximate ppl nsample"

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
        src_length = src.size(0)

        enc_final, memory_bank = self.encoder(src, lengths)
        enc_state = self.decoder.init_decoder_state(
            src, memory_bank, enc_final)
        if self.inference_network is not None and not self.no_q:
            # inference network q(z|x,y)
            q_scores = self.inference_network(src, inftgt, lengths, memory_bank) # batch_size, tgt_length, src_length
            #q_scores = self.inference_network(src, tgt, lengths, memory_bank) # batch_size, tgt_length, src_length
            q_scores_sample = sample_attn(scores=q_scores, dist_type=self.dist_type)
        else:
            q_scores, q_scores_sample = None, None

        decoder_outputs, dec_state, attns, p_a_scores = \
            self.decoder(tgt, memory_bank,
                         enc_state if dec_state is None
                         else dec_state,
                         memory_lengths=lengths,
                         q_scores_sample=q_scores_sample if not self.use_prior else None,
                         q_scores=q_scores if not self.use_prior else None)

        if self.approximate_ppl:
            sample_decoder_outputs = []
            for i in range(self.approximate_ppl_nsample):
                sample_decoder_output, _, _, _ = \
                self.decoder(tgt, memory_bank,
                            enc_state if dec_state is None
                            else dec_state,
                            memory_lengths=lengths,
                            q_scores_sample=None,
                            q_scores=q_scores if not self.use_prior else None)
                sample_decoder_outputs.append(sample_decoder_output)
            sample_decoder_outputs = torch.cat(sample_decoder_outputs, dim=0)
            # embed()?
        else:
            sample_decoder_outputs = None

        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None

        if self.inference_network is not None:
            for i in range(len(q_scores)):
                q_scores[i] = q_scores[i].view(batch_size, tgt_length, src_length)
            if self.dist_type == "dirichlet":
                return decoder_outputs, attns, dec_state,\
                   (q_scores[0], p_a_scores[0]), sample_decoder_outputs
            elif self.dist_type == "normal":
                return decoder_outputs, attns, dec_state,\
                   (q_scores[0], q_scores[1], p_a_scores[0], p_a_scores[1]), sample_decoder_outputs
            elif self.dist_type == "none":
                # sigh, lol. this whole thing needs to be cleaned up
                return decoder_outputs, attns, dec_state, (
                    q_scores_sample.transpose(0,1), p_a_scores[0]), sample_decoder_outputs
            else:
                raise Exception("Unsupported dist_type")
        else:
            return decoder_outputs, attns, dec_state, None, sample_decoder_outputs
        # p_a_scores: feed in sampled a, output unormalized attention scores (batch_size, tgt_length, src_length)
