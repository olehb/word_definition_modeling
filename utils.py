import torch
from torch import nn


def squash_packed(x, fn=torch.tanh):
    return torch.nn.utils.rnn.PackedSequence(fn(x.data), x.batch_sizes,
                 x.sorted_indices, x.unsorted_indices)


def init_data_loader() -> torch.utils.data.DataLoader:
    pass