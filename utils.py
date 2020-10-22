import numpy as np
import torch
from torch import nn
from wdm import Embeddings


def load_glove_embeddings(dim: int = 50, data_loc: str = './data/glove.6B/') -> dict:
    assert dim in [50, 100, 200, 300]

    embeddings = {}
    with open(f'{data_loc}/glove.6B.{dim}d.txt') as f:
        for line in f:
            elements = line.split(' ')
            assert len(elements) == dim+1
            word = elements[0]
            vector = torch.tensor(list(map(float, elements[1:])), dtype=torch.float32)
            embeddings[word] = vector

    embeddings[Embeddings.SOS] = torch.zeros(dim)
    embeddings[Embeddings.EOS] = torch.ones(dim)

    return Embeddings(embeddings)


def init_encoder() -> nn.Module:
    pass


def init_decoder() -> nn.Module:
    pass


def init_data_loader() -> torch.utils.data.DataLoader:
    pass