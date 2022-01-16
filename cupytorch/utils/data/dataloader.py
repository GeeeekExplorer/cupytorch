from typing import Tuple, Optional

import cupytorch as ct
from .dataset import Dataset


class DataLoader:

    def __init__(self, dataset: Dataset, batch_size: Optional[int] = 1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices = ct.randperm(len(self.dataset)) if shuffle else ct.arange(len(self.dataset))
        self.start = self.end = 0

    def __len__(self):
        return (len(self.dataset) - 1) // self.batch_size + 1

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[ct.Tensor, ...]:
        if self.end == len(self.dataset):
            self.end = 0
            raise StopIteration
        self.start = self.end
        self.end = min(self.start + self.batch_size, len(self.dataset))
        data = [self.dataset[index] for index in self.indices[self.start:self.end]]
        tensors = [ct.stack(t) for t in zip(*data)]
        return tuple(tensors)
