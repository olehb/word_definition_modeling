import torch
from torch import nn
from torch.nn import functional as F


class Embeddings:
    SOS = '<s>'
    EOS = '</s>'

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.dim = len(embeddings[self.SOS])

    def sentence_to_tensor(self, sentence):
        result = torch.empty(size=(len(sentence), self.dim), dtype=torch.float32)
        for i, word in enumerate(sentence):
            result[i] = self.embeddings[word]
        return result

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, item):
        return self.embeddings[item]


class LSTMEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

    def forward(self, batch):
        output, hidden = self.lstm(batch)
        # TODO: Activation function ReLU?
        return output, hidden


class LSTMDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        self.lstm = nn.LSTMCell(embedding_dim+hidden_dim, hidden_dim)
        self.vocab_proj = nn.Linear(hidden_dim, vocab_size)

