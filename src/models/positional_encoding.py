import torch
import numpy as np
from torch import nn
import math


class PositionalEncoding(torch.nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1, max_len: int = 15):
        """
        emb_size - размер эмбеддингов
        max_len - длинна контекста
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2) * (-math.log(10000.0)/emb_size))
        pe = torch.zeros(max_len, 1, emb_size)
        pe[:, 0, 0::2] = torch.sin(position*div_term)
        pe[:, 0, 1::2] = torch.cos(position*div_term)
        self.register_buffer('pe', pe)

    def forward(self, token_embedding):
        """
        token_embedding - тензор матрицы эмбеддингов
        """
        x = token_embedding + self.pe[:token_embedding.size(0)]
        return self.dropout(x)