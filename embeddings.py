import torch
from collections import defaultdict
import itertools


class Embeddings:
    SOS_STR = '<s>'
    EOS_STR = '</s>'
    UNK_STR = '<unk>'

    def __init__(self, embeddings, word2id, id2word, device):
        self.embeddings = embeddings
        self.dim = len(embeddings[self.SOS_STR])
        self.SOS = embeddings[self.SOS_STR]
        self.EOS = embeddings[self.EOS_STR]
        self.UNK = embeddings[self.UNK_STR]
        self.word2id = word2id
        self.id2word = id2word
        self.device = device

    def sentence_to_tensor(self, sentence):
        return torch.stack([self[word] for word in sentence])
        # return torch.cat([self[word] for word in sentence]).view(-1, 50)
        # return torch.cat([self[word] for word in sentence]).reshape(-1, 50)

    def sentence_to_ids(self, sentence):
        if type(sentence) not in (list, itertools.chain):
            sentence = sentence.split()
        return torch.tensor([self.word2id[word] if word in self.word2id else self.word2id[self.UNK_STR]
                             for word in sentence],
                            dtype=torch.long,
                            device=self.device)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, item):
        try:
            return self.embeddings[item]
        except KeyError:
            return self.UNK


def load_glove_embeddings(dim: int = 50, data_loc: str = './data/glove.6B/', device=None) -> dict:
    assert dim in [50, 100, 200, 300]

    embeddings = dict()
    embeddings[Embeddings.SOS_STR] = torch.zeros(dim, device=device)
    embeddings[Embeddings.EOS_STR] = torch.ones(dim, device=device)
    embeddings[Embeddings.UNK_STR] = -torch.ones(dim, device=device)

    word2id = dict()
    word2id[Embeddings.SOS_STR] = 0
    word2id[Embeddings.EOS_STR] = 1
    word2id[Embeddings.UNK_STR] = 2

    id2word = {v: k for k, v in word2id.items()}

    with open(f'{data_loc}/glove.6B.{dim}d.txt') as f:
        for line in f:
            elements = line.split(' ')
            assert len(elements) == dim+1
            word = elements[0]
            vector = torch.tensor(list(map(float, elements[1:])), dtype=torch.float32, device=device)
            embeddings[word] = vector
            word2id[word] = len(word2id)
            id2word[word2id[word]] = word  # Magic

    return Embeddings(embeddings, word2id, id2word, device)
