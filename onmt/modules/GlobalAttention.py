import torch
import torch.nn as nn

from torch.autograd import Function
from onmt.Utils import aeq, sequence_mask
import torch.nn.functional as F

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

class GlobalAttention(nn.Module):
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
          C --> F D --> F
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
    def __init__(self, dim, coverage=False, attn_type="dot", dist_type="normal"):
        super(GlobalAttention, self).__init__()

        self.dim = dim
        self.attn_type = attn_type
        self.dist_type = dist_type
        assert (self.attn_type in ["dot", "general", "mlp"]), (
                "Please select a valid attention type.")

        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)

        if self.dist_type == "normal":
            self.linear_1 = nn.Linear(dim + dim, 500)
            self.linear_2 = nn.Linear(500, 500)
            self.softplus = torch.nn.Softplus()
            self.tanh = torch.nn.Tanh()
            self.scale_out = nn.Linear(500,1)
            self.alpha_out = nn.Linear(500,1)
            #self.mean_out = nn.Linear(500, 1)
            #self.std_out = nn.Linear(500, 1)
            #self.bn_mu = nn.BatchNorm1d(1, affine=True)
            #self.bn_std = nn.BatchNorm1d(1, affine=True)
        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim*2, dim, bias=out_bias)

        self.sm = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

        if coverage:
            self.linear_cover = nn.Linear(1, dim, bias=False)

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

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch*tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)
        else:
            dim = self.dim
            wq = self.linear_query(h_t.view(-1, dim))
            wq = wq.view(tgt_batch, tgt_len, 1, dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

            uh = self.linear_context(h_s.contiguous().view(-1, dim))
            uh = uh.view(src_batch, 1, src_len, dim)
            uh = uh.expand(src_batch, tgt_len, src_len, dim)

            # (batch, t_len, s_len, d)
            wquh = self.tanh(wq + uh)

            return self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)

    def get_raw_scores(self, h_t, h_s, memory_lengths, align):
        """
            For log normal.
        """
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)
        aeq(self.dim, src_dim)
        
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
        ##h_std = self.softplus(self.bn_std(self.std_out(h_enc)))
        #h_alpha_log = self.softplus(self.scale_out(h_enc)) * self.tanh(self.alpha_out(h_enc))
        h_alpha_log = align # batch_size, tgt_len, src_len
        h_alpha_log = h_alpha_log.view(-1, src_len) #batch_size*tgt_len, src_len
        mask = sequence_mask(memory_lengths).view(-1,1,src_len).expand(-1,tgt_len,src_len).contiguous().view(-1,src_len) #batch_size, src_len
        memory_lengths_tgt = memory_lengths.view(-1, 1).float().expand(-1, tgt_len).contiguous().view(-1, 1)
        memory_lengths = memory_lengths.view(-1, 1, 1).float().expand(-1, tgt_len, src_len).contiguous().view(-1, src_len)
        h_alpha_log.data.masked_fill_(1-mask, 0)
        h_alpha_log_mean = h_alpha_log.sum(dim=-1, keepdim=True) / memory_lengths_tgt
        h_mean = h_alpha_log - h_alpha_log_mean
        k1 = 1 - 2. / memory_lengths
        k2 = 1. / (memory_lengths * memory_lengths)
        h_alpha = h_alpha_log.exp()
        h_alpha_inv = 1. / h_alpha
        h_alpha_inv.data.masked_fill_(1-mask, 0)
        h_std = k1 * h_alpha_inv + k2 * torch.sum(h_alpha_inv, dim=-1, keepdim=True)
        h_mean = h_mean.view(tgt_batch, tgt_len, src_len)
        h_std = h_std.view(tgt_batch, tgt_len, src_len)
        h_alpha = h_alpha.view(tgt_batch, tgt_len, src_len)
        #h_mean = self.mean_out(h_enc)
        #h_std = self.softplus(self.std_out(h_enc))
        #
        #h_mean = h_mean.view(tgt_batch, tgt_len, src_len)
        #h_std = h_std.view(tgt_batch, tgt_len, src_len)
        #h_mean, h_std = musigma(h_mean, h_std)
        #return [h_mean, h_std]
        return [h_mean, h_std, h_alpha, h_alpha_log.view(tgt_batch, tgt_len, src_len)]

    def forward(self, input, memory_bank, memory_lengths=None, coverage=None, q_scores_residual=None):
        """

        Args:
          input (`FloatTensor`): query vectors `[batch x tgt_len x dim]`
          memory_bank (`FloatTensor`): source vectors `[batch x src_len x dim]`
          memory_lengths (`LongTensor`): the source context lengths `[batch]`
          coverage (`FloatTensor`): None (not supported yet)

        Returns:
          (`FloatTensor`, `FloatTensor`):

          * Computed vector `[tgt_len x batch x dim]`
          * Attention distribtutions for each query
             `[tgt_len x batch x src_len]`
          * Unormalized attention scores for each query 
            `[batch x tgt_len x src_len]`
        """
        # one step input
        if input.dim() == 2:
            one_step = True
            input = input.unsqueeze(1)
            q_scores_residual = q_scores_residual.unsqueeze(1)
        else:
            assert (False)

        batch, sourceL, dim = memory_bank.size()
        tgt_batch = batch
        batch_, targetL, dim_ = input.size()
        src_len = sourceL
        tgt_len = targetL
        aeq(batch, batch_)
        aeq(dim, dim_)
        aeq(self.dim, dim)

        # compute attention scores, as in Luong et al.
        #align = self.score(input, memory_bank)
        align = self.score(input, memory_bank)

        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths)
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            align.data.masked_fill_(1 - mask, -999)

        # Softmax to normalize attention weights
        align = align.clamp(-10,10)
        if self.dist_type == "dirichlet":
            raw_scores = [align]
        elif self.dist_type == "normal":
            raw_scores = self.get_raw_scores(input, memory_bank, memory_lengths, align)
        else:
            raw_scores = [align]
        align_vectors = self.sm(align.view(batch*targetL, sourceL))
        align_vectors = align_vectors.view(batch, targetL, sourceL)

        # each context vector c_t is the weighted average
        # over all the source hidden states
                    #q_scores_sample = F.softmax(m.rsample().cuda(), dim=-1).view(batch_size, tgt_length, -1).transpose(0,1)
        mask = sequence_mask(memory_lengths)
        mask = mask.unsqueeze(1)  # Make it broadcastable.
        raw_scores[0].data.masked_fill_(1 - mask, -999)
        raw_scores[1].data.masked_fill_(1 - mask, 0.001)
        raw_scores[2].data.masked_fill_(1 - mask, 0.001)
        raw_scores[3].data.masked_fill_(1 - mask, -999)

        memory_lengths_orig = memory_lengths
        q_scores_log = q_scores_residual + raw_scores[3]
        q_scores_log = q_scores_log.clamp(-10, 10)
        q_scores_log = q_scores_log.view(-1, src_len) #batch_size*tgt_len, src_len
        mask = sequence_mask(memory_lengths).view(-1,1,src_len).expand(-1,tgt_len,src_len).contiguous().view(-1,src_len) #batch_size, src_len
        memory_lengths_tgt = memory_lengths.view(-1, 1).float().expand(-1, tgt_len).contiguous().view(-1, 1)
        memory_lengths = memory_lengths.view(-1, 1, 1).float().expand(-1, tgt_len, src_len).contiguous().view(-1, src_len)
        q_scores_log.data.masked_fill_(1-mask, 0)
        q_scores_log_mean = q_scores_log.sum(dim=-1, keepdim=True) / memory_lengths_tgt
        q_scores_mean = q_scores_log - q_scores_log_mean
        k1 = 1 - 2. / memory_lengths
        k2 = 1. / (memory_lengths * memory_lengths)
        q_scores_alpha = q_scores_log.exp()
        q_scores_alpha_inv = 1. / q_scores_alpha
        q_scores_alpha_inv.data.masked_fill_(1-mask, 0)
        q_scores_std = k1 * q_scores_alpha_inv + k2 * torch.sum(q_scores_alpha_inv, dim=-1, keepdim=True)
        q_scores_mean = q_scores_mean.view(tgt_batch, tgt_len, src_len)
        q_scores_std = q_scores_std.view(tgt_batch, tgt_len, src_len)
        q_scores_alpha = q_scores_alpha.view(tgt_batch, tgt_len, src_len)
        mask = sequence_mask(memory_lengths_orig)
        mask = mask.unsqueeze(1)  # Make it broadcastable.
        q_scores_mean.data.masked_fill_(1 - mask, -999)
        q_scores_std.data.masked_fill_(1 - mask, 0.001)
        q_scores_alpha.data.masked_fill_(1 - mask, 0.001)
        #m = torch.distributions.normal.Normal(raw_scores[0].view(-1, sourceL), raw_scores[1].view(-1, sourceL))
        #p_a_scores_sample = F.softmax(m.rsample().cuda(), dim=-1).view(batch, targetL, sourceL)
        #p_a_scores_sample = raw_scores[0].view(batch, targetL, sourceL)
        m = torch.distributions.normal.Normal(q_scores_mean.view(-1, sourceL), q_scores_std.view(-1, sourceL))
        q_scores_sample = F.softmax(m.rsample().cuda(), dim=-1).view(batch, targetL, sourceL)
        #c = torch.bmm(p_a_scores_sample, memory_bank)
        c = torch.bmm(q_scores_sample, memory_bank)
        # what is size of q_scores_sample? batch, targetL, sourceL

        # concatenate
        concat_c = torch.cat([c, input], 2).view(batch*targetL, dim*2)
        attn_h = self.linear_out(concat_c).view(batch, targetL, dim)
        if self.attn_type in ["general", "dot"]:
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
            assert (False)
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

        return attn_h, align_vectors, raw_scores, (q_scores_mean, q_scores_std, q_scores_alpha, q_scores_sample)
