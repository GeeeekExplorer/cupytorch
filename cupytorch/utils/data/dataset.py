from bisect import bisect_right
from itertools import accumulate
from typing import Collection, Any, Tuple

from cupytorch import Tensor


class Dataset:

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index) -> Any:
        raise NotImplementedError


class TensorDataset(Dataset):

    def __init__(self, *tensors: Tensor) -> None:
        self.tensors = tensors

    def __len__(self) -> int:
        return len(self.tensors[0])

    def __getitem__(self, index) -> Tuple[Tensor, ...]:
        return tuple(tensor[index] for tensor in self.tensors)


class ConcatDataset(Dataset):

    def __init__(self, datasets: Collection[Dataset]) -> None:
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, "datasets should not be an empty iterable"
        self.datasets = list(datasets)
        self.cumulative_sizes = list(accumulate(len(d) for d in self.datasets))

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx) -> Any:
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]
