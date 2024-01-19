import numpy as np
import pandas as pd

import math
import torch
import torch.nn as nn

# 1. Input Embedding 
class InputEmbedding(nn.Module): 
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

# 2. Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout:float):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.pe = torch.zeros(self.seq_len, self.d_model)
        position = torch.arange(0, self.seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * -(math.log(10000.0) / self.d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term) ; self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)
        
    def forward(self, x):
        x = x + self.pe[:, :x.shape[1],:].requires_grad_(False)
        return self.dropout(x)

# 3. Layer Normalization (eps set to 1e-6)
class LayerNormalization(nn.Module):
    def __init__(self, eps= 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

# 4. Feed Forward (d_ff: 2048)
class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.d_ff = d_ff
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.layer_1 = nn.Linear(self.d_model, self.d_ff)
        self.layer_2 = nn.Linear(self.d_ff, self.d_model)
    def forward(self, x):
        return self.layer_2(self.dropout(torch.relu((self.layer_1(x)))))

# 5. Multi Head Attention (h:8)
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, h, dropout):
        super().__init__()
        self.h = h
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        assert self.d_model / self.h
        self.d_k = self.d_model // self.h
        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)
        self.w_o = nn.Linear(self.d_model, self.d_model)

    # calculate attention score
    def attention(self, query, key, value, mask):
        attention_scores = (query @ key.transpose(-1, -2)) / math.sqrt(self.d_k)
        if mask is not None:
            attention_scores.masked_fill(mask ==0, 1e-9)
        attention_scores = attention_scores.softmax(dim = -1)
        # print(attention_scores.shape)
        attention_scores = self.dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
        query = q.view(q.shape[0], q.shape[1], self.h, self.d_k).transpose(1, 2) # change sequence
        key = k.view(k.shape[0], k.shape[1], self.h, self.d_k).transpose(1, 2)
        value = v.view(v.shape[0], v.shape[1], self.h, self.d_k).transpose(1, 2)
        attention_scores, x = self.attention(query, key, value, mask)
        attention_scores = attention_scores.transpose(1, 2).contiguous().view(attention_scores.shape[0], -1, self.h * self.d_k)
        return self.w_o(attention_scores)

# 6. Residual Connection (dropout, normalization only applied to the sublayer)
class ResidualConnection(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

# 6. Encoder Block (two Residual Connectuions)
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardLayer, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

# 7. Encoder (n_layer: 6)
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
