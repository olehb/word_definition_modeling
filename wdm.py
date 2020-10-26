import torch
from torch import nn
from utils import squash_packed


class LSTMEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        # TODO: Add dropout to LSTM
        # self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=0.2, bidirectional=True)

    def forward(self, batch):
        output, (hidden, _) = self.lstm(batch)
        # TODO: Activation function ReLU?
        return output, hidden


class LSTMCellDecoder(nn.Module):
    def __init__(self, dim, hidden_dim, vocab_size):
        super().__init__()
        self.lstm = nn.LSTM(dim, hidden_dim)
        self.vocab_proj = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim

    def forward(self, input, hidden):
        output, (_, _) = self.lstm(input, (hidden, torch.zeros(size=hidden.shape)))
        # TODO: Activation function ReLU?
        output = squash_packed(output, self.vocab_proj)
        return output
