import torch
from torch.nn.utils.rnn import PackedSequence


def squash_packed(x, fn=torch.tanh):
    return PackedSequence(fn(x.data), x.batch_sizes, x.sorted_indices, x.unsorted_indices)

