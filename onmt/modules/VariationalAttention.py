import torch
import torch.nn as nn
import torch.nn.functional as F

from onmt.Utils import aeq, sequence_mask


class VariationalAttention(nn.Module):
    """
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.

    Constructs a unit mapping a query `q` of size `dim`
    and a source matrix `H` of size `n x dim`, to an output
    of size `dim`.


    .. mermaid::

       graph BT
          A[Query]
          subgraph RNN
            C[H 1]
            D[H 2]
            E[H N]
          end
          F[Attn]
          G[Output]
          A --> F
          C --> F
          D --> F
          E --> F
          C -.-> G
          D -.-> G
          E -.-> G
          F --> G

    All models compute the output as
    :math:`c = \sum_{j=1}^{SeqLength} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].

    However they
    differ on how they compute the attention score.

    * Luong Attention (dot, general):
       * dot: :math:`score(H_j,q) = H_j^T q`
       * general: :math:`score(H_j, q) = H_j^T W_a q`


    * Bahdanau Attention (mlp):
       * :math:`score(H_j, q) = v_a^T tanh(W_a q + U_a h_j)`


    Args:
       dim (int): dimensionality of query and key
       coverage (bool): use coverage term
       attn_type (str): type of attention to use, options [dot,general,mlp]

    """
    def __init__(
        self, dim,
        dist_type="log_normal",
        use_prior=False
    ):
        super(VariationalAttention, self).__init__()

        # attn type is always general
        # no coverage crap
        self.dim = dim
        self.dist_type = dist_type
        self.use_prior = use_prior

        self.linear_in = nn.Linear(dim, dim, bias=False)

        if self.dist_type == "log_normal":
            self.linear_1 = nn.Linear(dim + dim, 100)
            self.linear_2 = nn.Linear(100, 100)
            self.softplus = torch.nn.Softplus()
            self.mean_out = nn.Linear(100, 1)
            self.var_out = nn.Linear(100, 1)
        # mlp wants it with bias
        out_bias = False
        self.linear_out = nn.Linear(dim*2, dim, bias=out_bias)

        self.sm = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def score(self, h_t, h_s):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x tgt_len x dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`

        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`

        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)
        aeq(self.dim, src_dim)

        h_t_ = h_t.view(tgt_batch*tgt_len, tgt_dim)
        h_t_ = self.linear_in(h_t_)
        h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
        h_s_ = h_s.transpose(1, 2)
        # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
        return torch.bmm(h_t, h_s_)

    def get_raw_scores(self, h_t, h_s):
        """
            For log normal.
        """
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)
        aeq(self.dim, src_dim)
        
        h_t_expand = h_t.unsqueeze(2).expand(-1, -1, src_len, -1)
        h_s_expand = h_s.unsqueeze(1).expand(-1, tgt_len, -1, -1)
        # [batch, tgt_len, src_len, src_dim]
        h_expand = torch.cat((h_t_expand, h_s_expand), dim=3)
        h_fold = h_expand.contiguous().view(-1, src_dim + tgt_dim)
        
        h_enc = self.softplus(self.linear_1(h_fold))
        h_enc = self.softplus(self.linear_2(h_enc))
        
        h_mean = self.softplus(self.mean_out(h_enc))
        h_var = self.softplus(self.var_out(h_enc))
        
        h_mean = h_mean.view(tgt_batch, tgt_len, src_len)
        h_var = h_var.view(tgt_batch, tgt_len, src_len)
        return [h_mean, h_var]

    def forward(self, input, memory_bank, memory_lengths=None, coverage=None, q_scores_sample=None):
        """

        Args:
          input (`FloatTensor`): query vectors `[batch x tgt_len x dim]`
          memory_bank (`FloatTensor`): source vectors `[batch x src_len x dim]`
          memory_lengths (`LongTensor`): the source context lengths `[batch]`
          coverage (`FloatTensor`): None (not supported yet)

        Returns:
          (`FloatTensor`, `FloatTensor`):

          * Weighted context vector `[tgt_len x batch x dim]`
          * Attention distribtutions for each query
             `[tgt_len x batch x src_len]`
          * Unormalized attention scores for each query 
            `[batch x tgt_len x src_len]`
        """

        # one step input
        if input.dim() == 2:
            one_step = True
            input = input.unsqueeze(1)
            if q_scores_sample is not None:
                q_scores_sample = q_scores_sample.unsqueeze(1)
        else:
            one_step = False

        batch, sourceL, dim = memory_bank.size()
        batch_, targetL, dim_ = input.size()
        aeq(batch, batch_)
        aeq(dim, dim_)
        aeq(self.dim, dim)

        # compute attention scores, as in Luong et al.
        #align = self.score(input, memory_bank)
        align = self.score(input, memory_bank)

        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths)
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            align.data.masked_fill_(1 - mask, -float('inf'))

        # Softmax to normalize attention weights
        if self.dist_type == "dirichlet":
            align = align.clamp(1e-2, 5).exp()
            raw_scores = [align]
        elif self.dist_type == "log_normal":
            raw_scores = self.get_raw_scores(input, memory_bank)
        else:
            raw_scores = [align]

        if q_scores_sample is None or self.dist_type == "none":
            align_vectors = self.sm(align.view(batch*targetL, sourceL))
            align_vectors = align_vectors.view(batch, targetL, sourceL)
        else:
            # sample from the prior
            m = torch.distributions.Dirichlet(
                align
                    #.masked_fill(1-mask, float("-inf"))
                    .view(batch*targetL, sourceL)
                    .cpu()
            )
            align_vectors = m.rsample().cuda(align.get_device()).view(batch, targetL, -1)
            # if we're zeroing out the approximate posterior, wait what?
            #align_vectors = align_vectors.masked_fill(1-mask, 0)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        if q_scores_sample is None or self.use_prior:
            c = torch.bmm(align_vectors, memory_bank)
        else:
            c = torch.bmm(q_scores_sample, memory_bank)
        # what is size of q_scores_sample? batch, targetL, sourceL

        # concatenate
        concat_c = torch.cat([c, input], 2).view(batch*targetL, dim*2)
        attn_h = self.linear_out(concat_c).view(batch, targetL, dim)
        attn_h = self.tanh(attn_h)

        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)

            # Check output sizes
            batch_, dim_ = attn_h.size()
            aeq(batch, batch_)
            aeq(dim, dim_)
            batch_, sourceL_ = align_vectors.size()
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)
        else:
            attn_h = attn_h.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()

            # Check output sizes
            targetL_, batch_, dim_ = attn_h.size()
            aeq(targetL, targetL_)
            aeq(batch, batch_)
            aeq(dim, dim_)
            targetL_, batch_, sourceL_ = align_vectors.size()
            aeq(targetL, targetL_)
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)

        # attn_h: convex combination of memory_bank
        # align_vectors: convex coefficients / boltzmann dist
        # raw_scores: unnormalized scores
        return attn_h, align_vectors, raw_scores

    # @overload
    def train(self, mode=True):
        # use the generative model during evaluation (mode=False)
        self.use_prior = not mode
        super(VariationalAttention, self).train(mode)

