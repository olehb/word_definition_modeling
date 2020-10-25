import torch
from collections import defaultdict


class Embeddings:
    SOS_STR = '<s>'
    EOS_STR = '</s>'

    def __init__(self, embeddings, word2id, id2word):
        self.embeddings = embeddings
        self.dim = len(embeddings[self.SOS_STR])
        self.SOS = embeddings[self.SOS_STR]
        self.EOS = embeddings[self.EOS_STR]
        self.word2id = word2id
        self.id2word = id2word

    def sentence_to_tensor(self, sentence):
        result = torch.empty(size=(len(sentence), self.dim), dtype=torch.float32)
        for i, word in enumerate(sentence):
            result[i] = self.embeddings[word]
        return result

    def sentence_to_ids(self, sentence):
        if not type(sentence) == list:
            sentence = sentence.split()
        return torch.tensor([self.word2id[word] for word in sentence], dtype=torch.int32)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, item):
        return self.embeddings[item]


def load_glove_embeddings(dim: int = 50, data_loc: str = './data/glove.6B/') -> dict:
    assert dim in [50, 100, 200, 300]

    embeddings = dict()
    embeddings[Embeddings.SOS_STR] = torch.zeros(dim)
    embeddings[Embeddings.EOS_STR] = torch.ones(dim)

    word2id = defaultdict(lambda: len(word2id))
    word2id[Embeddings.SOS_STR] = 0
    word2id[Embeddings.EOS_STR] = 1

    id2word = {v: k for v, k in word2id.items()}

    with open(f'{data_loc}/glove.6B.{dim}d.txt') as f:
        for line in f:
            elements = line.split(' ')
            assert len(elements) == dim+1
            word = elements[0]
            vector = torch.tensor(list(map(float, elements[1:])), dtype=torch.float32)
            embeddings[word] = vector
            id2word[word2id[word]] = word  # Magic

    return Embeddings(embeddings, word2id, id2word)
