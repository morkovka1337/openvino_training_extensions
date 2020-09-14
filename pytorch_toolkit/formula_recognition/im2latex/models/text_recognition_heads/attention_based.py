import math
from collections import namedtuple, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from ...data.vocab import END_TOKEN, START_TOKEN

INIT = 1e-2
Candidate = namedtuple('candidate', 'score, dec_state_h, dec_state_c, output, targets, logits')


class TextRecognitionHead(nn.Module):
    def __init__(self, out_size, configuration):
        super(TextRecognitionHead, self).__init__()
        emb_size = configuration.get("emb_size")
        enc_rnn_h = configuration.get("enc_rnn_h")
        dec_rnn_h = configuration.get("dec_rnn_h")
        max_len = configuration.get("max_len")
        n_layer = configuration.get("n_layer")
        beam_width = configuration.get('beam_width')
        in_lstm_ch = configuration.get('in_lstm_ch')
        self.enc_rnn_h = enc_rnn_h
        self.dec_rnn_h = dec_rnn_h
        self.emb_dim = emb_size
        self.out_size = out_size

        self.max_len = max_len
        self.beam_width = beam_width
        self.rnn_encoder = nn.LSTM(in_lstm_ch, enc_rnn_h,
                                   bidirectional=True,
                                   batch_first=True)
        self.rnn_decoder = nn.LSTMCell(enc_rnn_h+emb_size, dec_rnn_h)
        self.embedding = nn.Embedding(out_size, emb_size)

        # enc_rnn_h*2 is the dimension of context
        self.W_c = nn.Linear(dec_rnn_h+2*enc_rnn_h, enc_rnn_h)
        self.W_out = nn.Linear(enc_rnn_h, out_size)

        # a trainable initial hidden state V_h_0 for each row
        self.V_h_0 = nn.Parameter(torch.Tensor(n_layer*2, enc_rnn_h))
        self.V_c_0 = nn.Parameter(torch.Tensor(n_layer*2, enc_rnn_h))
        init.uniform_(self.V_h_0, -INIT, INIT)
        init.uniform_(self.V_c_0, -INIT, INIT)

        # Attention mechanism
        self.beta = nn.Parameter(torch.Tensor(dec_rnn_h))
        init.uniform_(self.beta, -INIT, INIT)
        self.W_h = nn.Linear(dec_rnn_h, dec_rnn_h)
        self.W_v = nn.Linear(enc_rnn_h*2, dec_rnn_h)

    def forward(self, features, formulas=None):
        """args:
        imgs: [B, C, H, W]
        formulas: [B, MAX_LEN]

        return:
        logits: [B, MAX_LEN, VOCAB_SIZE]
        """
        # encoding
        row_enc_out, hidden, context = self.encode(features)
        # init decoder's states
        h, c, O_t = self.init_decoder(row_enc_out, hidden, context)

        if formulas is not None:
            logits, targets = self.decode_with_formulas(
                h, c, O_t, formulas, row_enc_out)
        elif self.beam_width == 0:
            device = features.device
            b_size = features.size(0)
            logits, targets = self.decode_without_formulas(
                h, c, O_t, row_enc_out, b_size, device)
        else:
            assert features.size(
                0) == 1, "Using beam search, batch size must be equal to zero"
            device = features.device
            b_size = features.size(0)
            logits, targets = self.decode_with_bs(
                h, c, O_t, row_enc_out, b_size, device)

        return logits, targets

    def decode_with_formulas(self, h, c, O_t, formulas, row_enc_out):
        logits = []
        # do not use beam search; predict as usual
        max_len = formulas.size(1)
        for t in range(max_len):
            tgt = formulas[:, t:t+1]
            h, c, O_t, logit = self.step_decoding(
                h, c, O_t, row_enc_out, tgt)
            logits.append(logit)
        logits = torch.stack(logits, dim=1)  # [B, MAX_LEN, out_size]
        targets = torch.max(torch.log(logits).data, dim=2)[1]
        return logits, targets

    def decode_without_formulas(self, h, c, output, row_enc_out, b_size, device):
        logits = []
        logit = None
        for _ in range(self.max_len):
            if logit is not None:
                tgt = torch.reshape(torch.max(logit.data, dim=1)[
                                    1], (b_size, 1)).clone().detach()
            else:
                tgt = torch.tensor([[START_TOKEN]] * b_size)
            tgt = tgt.to(device)
            # one step decoding
            h, c, output, logit = self.step_decoding(
                h, c, output, row_enc_out, tgt)

            logits.append(logit)
        logits = torch.stack(logits, dim=1)  # [B, MAX_LEN, out_size]
        targets = torch.max(torch.log(logits).data, dim=2)[1]
        return logits, targets

    def decode_with_bs(self, h, c, O_t, row_enc_out, b_size, device):
        max_len = self.max_len
        # initializing empty list of candidates
        # at first we have only one candidate (START_TOKEN) per image with score = 1
        candidates_list = []
        candidates_list.append(Candidate(
            score=0.,
            dec_state=(h, c),
            Out_t=O_t,
            targets=[START_TOKEN],
            logits=[]
        ))
        # the first prediction is empty with score = 1
        logits = [[]] * b_size
        targets = [[]] * b_size

        for _ in range(max_len):
            # pushing every candidate into step decoding to get probable tracks
            all_candidates = self.decode_with_bs_step(
                row_enc_out, candidates_list, device)
            all_candidates.sort(key=lambda cand: cand.score)
            candidates_list = all_candidates[:self.beam_width]

        logits = candidates_list[0].logits
        targets = candidates_list[0].targets[1:]
        logits = torch.stack(logits, dim=0)
        logits = logits.unsqueeze(dim=0)
        logits = logits.to(device)
        targets = torch.tensor(targets).to(device)
        targets = targets.unsqueeze(dim=0)

        return logits, targets

    def encode(self, encoded_imgs):
        encoded_imgs = encoded_imgs.permute(0, 2, 3, 1)  # [B, H', W', LSTM_INP_CHANNELS]
        # Prepare data for Row Encoder
        # poccess data like a new big batch
        B, H, W, out_channels = encoded_imgs.size()

        encoded_imgs = encoded_imgs.contiguous().view(B*H, W, out_channels)

        # prepare init hidden for each row
        init_hidden_h = self.V_h_0.unsqueeze(
            1).expand(self.V_h_0.shape[0], B*H, self.V_h_0.shape[1]).contiguous()
        init_hidden_c = self.V_c_0.unsqueeze(
            1).expand(self.V_h_0.shape[0], B*H, self.V_h_0.shape[1]).contiguous()
        init_hidden = (init_hidden_h, init_hidden_c)

        # Row Encoder
        self.rnn_encoder.flatten_parameters()
        row_enc_out, (h, c) = self.rnn_encoder(encoded_imgs, init_hidden)
        # row_enc_out [B*H, W, enc_rnn_h]
        # hidden: [2, B*H, enc_rnn_h]
        row_enc_out = row_enc_out.view(B, H, W, self.dec_rnn_h)  # [B, H, W, dec_rnn_h]
        h = h.view(2, B, H, self.enc_rnn_h)
        c = c.view(2, B, H, self.enc_rnn_h)
        return row_enc_out, h, c

    def step_decoding(self, h, c, output, enc_out, tgt):
        """Runing one step decoding"""

        prev_y = self.embedding(tgt).squeeze(1)  # [B, emb_size]
        inp = torch.cat([prev_y, output], dim=1)  # [B, emb_size+enc_rnn_h]
        h_t, c_t = self.rnn_decoder(inp, (h, c))

        context_t, attn_scores = self._get_attn(enc_out, h)
        # [B, enc_rnn_h]
        output = self.W_c(torch.cat([h_t, context_t], dim=1)).tanh()

        # calculate logit
        logit = F.softmax(self.W_out(output), dim=1)  # [B, out_size]

        return h_t, c_t, output, logit

    def decode_with_bs_step(self, enc_out, candidates_list, device):
        all_candidates = []

        for cand in candidates_list:
            *cur_dec_st, cur_out, cur_logit = self.step_decoding(
                h=cand.dec_state_h,  # unique for unique image
                c=cand.dec_state_c,
                output=cand.output,  # the same
                enc_out=enc_out,
                tgt=torch.tensor([[cand.targets[-1]]]).to(device)
            )
            if cand.targets[-1] == END_TOKEN:
                all_candidates.append(cand)
                continue
            for k in range(cur_logit.size(1)):
                new_track = Candidate(
                    score=- math.log(cur_logit[:, k:k+1].item()) + cand.score,
                    dec_state_h=cur_dec_st[0],
                    dec_state_c=cur_dec_st[1],
                    output=cur_out,
                    targets=cand.targets + [k],
                    logits=cand.logits + [cur_logit[0].data]
                )
                all_candidates.append(new_track)

        return all_candidates

    def _get_attn(self, enc_out, prev_h):
        """Attention mechanism
        args:
            enc_out: row encoder's output [B, H, W, enc_rnn_h]
            prev_h: the previous time step hidden state [B, dec_rnn_h]
        return:
            context: this time step context [B, enc_rnn_h]
            attn_scores: Attention scores
        """
        # self.W_v(enc_out) [B, H, W, enc_rnn_h]
        # self.W_h(prev_h) [B, enc_rnn_h]
        B, H, W, _ = enc_out.size()
        linear_prev_h = self.W_h(prev_h)
        lin_pr_sh = linear_prev_h.shape
        linear_prev_h = linear_prev_h.view(B, 1, 1, lin_pr_sh[-1])
        linear_prev_h = linear_prev_h.expand(lin_pr_sh[0], H, W, lin_pr_sh[-1])
        e = torch.sum(self.beta * torch.tanh(
            linear_prev_h + self.W_v(enc_out)
        ),
            dim=-1
        )  # [B, H, W]

        alpha = F.softmax(e.view(B, -1), dim=-1).view(B, H, W)
        attn_scores = alpha.unsqueeze(-1)
        context = torch.sum(attn_scores * enc_out,
                            dim=[1, 2])  # [B, enc_rnn_h]

        return context, attn_scores

    def init_decoder(self, enc_out, hidden, context):
        """args:
            enc_out: the output of row encoder [B, H, W, enc_rnn_h]
            hidden: the last step hidden of row encoder [2, B, H, enc_rnn_h]
          return:
            h_0, c_0  h_0 and c_0's shape: [B, dec_rnn_h]
            init_O : the average of enc_out  [B, enc_rnn_h]
            for decoder
        """
        hidden, context = self._convert_hidden(hidden), self._convert_hidden(context)
        context_0 = enc_out.mean(dim=[1, 2])
        concat = torch.cat([hidden, context_0], dim=1)
        w_c_result = self.W_c(concat)
        init_0 = torch.tanh(w_c_result)
        return hidden, context, init_0

    def _convert_hidden(self, hidden):
        """convert row encoder hidden to decoder initial hidden"""
        hidden = hidden.permute(1, 2, 0, 3).contiguous()
        # Note that 2*enc_rnn_h = dec_rnn_h
        hidden = hidden.view(hidden.size(
            0), hidden.size(1), self.dec_rnn_h)  # [B, H, dec_rnn_h]
        hidden = hidden.mean(dim=1)  # [B, dec_rnn_h]

        return hidden