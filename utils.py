import numpy as np
import torch
from torch import nn


def load_glove_embeddings(dim: int = 50, data_loc: str = './data/glove.6B/') -> dict:
    assert dim in [50, 100, 200, 300]
    embeddings = {}
    with open(f'{data_loc}/glove.6B.{dim}d.txt') as f:
        for line in f:
            elements = line.split(' ')
            assert len(elements) == dim+1
            word = elements[0]
            vector = np.asarray(elements[1:], dtype=np.float32)
            embeddings[word] = vector
    return embeddings


def init_encoder() -> nn.Module:
    pass


def init_decoder() -> nn.Module:
    pass


def init_data_loader() -> torch.utils.data.DataLoader:
    pass