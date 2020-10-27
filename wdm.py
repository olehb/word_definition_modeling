from typing import Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence

from utils import squash_packed


class LSTMEncoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        # TODO: Add dropout to LSTM
        # self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=0.2, bidirectional=True)

    def forward(self, batch: PackedSequence) -> Tuple[PackedSequence, torch.Tensor]:
        output, (hidden, _) = self.lstm(batch)
        # TODO: Activation function ReLU?
        return output, hidden


class LSTMCellDecoder(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.lstm = nn.LSTM(dim, hidden_dim)
        self.vocab_proj = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim

    def forward(self, input: PackedSequence, hidden: torch.Tensor) -> PackedSequence:
        output, (_, _) = self.lstm(input, (hidden, torch.zeros(size=hidden.shape).to(hidden.device)))
        # TODO: Activation function ReLU?
        output = squash_packed(output, self.vocab_proj)
        return output
