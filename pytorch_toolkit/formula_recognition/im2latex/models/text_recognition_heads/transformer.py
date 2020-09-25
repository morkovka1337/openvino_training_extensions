import math
import torch.nn as nn
import torch
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=192):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # print(pe.shape)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, out_size, head_config):
        super().__init__()
        in_channels = head_config.get('in_channels')
        d_model = head_config.get('d_model', 512)
        pool_dim = head_config.get('pool_dim', 12)
        conv_dim = head_config.get('conv_dim', 12)
        emb_size = head_config.get('emb_size')
        dropout = head_config.get('dropout', 0.2)
        n_head = head_config.get('n_head', 200)
        n_token = out_size
        n_layers = head_config.get('n_layers', 2)
        n_hid = head_config.get('n_hid', 200)
        self.d_model = d_model
        # self.pool = nn.AdaptiveAvgPool2d(pool_dim)
        # self.conv = nn.Conv2d(in_channels=in_channels, out_channels=d_model, kernel_size=(1, conv_dim))
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_head, n_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.encoder = nn.Embedding(n_token, d_model)
        self.d_model = d_model
        self.src_mask = None
        self.decoder = nn.Linear(d_model, n_token)
        # self.transformer = nn.Transformer(d_model=d_model)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, encoded_imgs, tgt=None):
        # x = self.pool(encoded_imgs)
        # x = self.conv(x)
        # [B, C, H, W]
        N, C, H, W = encoded_imgs.shape
        # x = encoded_imgs.contiguous().permute(2, 3, 0, 1).view(W*H, N, C)
        encoded_imgs = encoded_imgs.contiguous().permute(3, 0, 1, 2).view(W, N, C*H)
        device = encoded_imgs.device
        if self.src_mask is None or self.src_mask.size(0) != len(encoded_imgs):
            mask = self._generate_square_subsequent_mask(len(encoded_imgs)).to(device)
            self.src_mask = mask

        # src = self.encoder(encoded_imgs.to(torch.long)) * math.sqrt(self.emb_size)

        src = self.pos_encoder(encoded_imgs)
        # print(src.shape)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        # return F.log_softmax(output, dim=-1)
        # if tgt is not None:
        #     tgt = self.preprocess_tgt(tgt)
        # else:
        #     tgt = torch.zeros(encoded_imgs.size(0), encoded_imgs.size(1), self.d_model)
        #     tgt = tgt.permute(1, 0, 2).to(encoded_imgs.device)
        # res = self.transformer(x, tgt)
        return self.postprocess_out(output)

    def preprocess_tgt(self, tgt):
        tgt_pad = torch.zeros(tgt.size(0), tgt.size(1), self.d_model)
        # tgt_pad shape: [Length of the formula, batch size, d_model (dimension of the vocab)]
        # tgt: [5, 2, 5] -> [[0, 0, 0, 0, 1, 0, 0...], [0, 0, 1, 0, ...],...]
        for example in range(tgt.shape[0]):  # iterating over batches
            for idx in range(len(tgt[example])):
                tgt_pad[example, idx, tgt[example, idx]] = 1
        tgt_pad = tgt_pad.permute(1, 0, 2)
        return tgt_pad.to(tgt.device)

    def postprocess_out(self, output):
        logits = F.softmax(output, dim=2)
        logits = logits.permute(1, 0, 2)
        targets = torch.max(torch.log(logits).data, dim=2)[1]
        return logits, targets
