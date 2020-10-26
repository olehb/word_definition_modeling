import torch
from torch.utils.data.dataset import Dataset


class Oxford2019Dataset(Dataset):
    def __init__(self, data_loc='./data/Oxford-2019/ALL.txt'):
        self.items = []

        errors = 0
        with open(data_loc) as f:
            for line in f:
                elements = line.split('|||')
                word = elements[0]
                definition = elements[3]
                example = elements[4]

                # if word not in example:
                #     errors += 1
                #     print(f'"{word}" not in "{example}"')
                self.items.append([word, definition, example])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def squash_packed(x, fn=torch.tanh):
    return torch.nn.utils.rnn.PackedSequence(fn(x.data), x.batch_sizes,
                 x.sorted_indices, x.unsorted_indices)
