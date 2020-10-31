from torch.utils.data.dataset import Dataset
from typing import Tuple


class Oxford2019Dataset(Dataset):
    def __init__(self, data_loc: str):
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
                self.items.append((word, ' '.join([word, example]), definition, example))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[str, str, str]:
        return self.items[idx]
