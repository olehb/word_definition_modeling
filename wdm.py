import torch
from torch import nn
from torch.nn import functional as F


class Embeddings:
    SOS_STR = '<s>'
    EOS_STR = '</s>'

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.dim = len(embeddings[self.SOS_STR])
        self.SOS = embeddings[self.SOS_STR]
        self.EOS = embeddings[self.EOS_STR]

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
        super().__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        # self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=0.2, bidirectional=True)

    def forward(self, batch):
        output, hidden = self.lstm(batch)
        # TODO: Activation function ReLU?
        return output, hidden


class LSTMCellDecoder(nn.Module):
    def __init__(self, max_length, hidden_dim, vocab_size):
        super().__init__()
        self.max_length = max_length
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)
        self.vocab_proj = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim

    def forward(self, context):
        result = []
        hidden = torch.zeros(size=(4, 100))
        cell = torch.zeros(size=(4, 100))

        for i in range(self.max_length):
            hidden, cell = self.lstm(context, (hidden, cell))

            out = F.relu(hidden)
            logits = self.vocab_proj(out)
            proj = F.log_softmax(logits)
            _, top_id = proj.topk(1)
            result.append(top_id)

        return result

