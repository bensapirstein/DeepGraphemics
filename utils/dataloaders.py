import math
import numpy as np
import torch
import torch.utils.data
from typing import Sized, Iterator
from torch.utils.data import Dataset, Sampler, SequentialSampler


class FirstLastSampler(Sampler):
    """
    A sampler that returns elements in a first-last order.
    """

    def __init__(self, data_source: Sized):
        """
        :param data_source: Source of data, can be anything that has a len(),
        since we only care about its number of elements.
        """
        super().__init__(data_source)
        self.data_source = data_source

        # Alternative implementation:
        # N = len(data_source)
        # first_half = range(N // 2)
        # last_half = range(N-1, N // 2 - 1, -1)
        # self.indices = [i for pair in zip(first_half, last_half) for i in pair]
        # # handle odd number of samples
        # if N % 2 == 1:
        #     self.indices.append(N // 2)

    def __iter__(self) -> Iterator[int]:
        # TODO:
        # Implement the logic required for this sampler.
        # If the length of the data source is N, you should return indices in a
        # first-last ordering, i.e. [0, N-1, 1, N-2, ...].
        direction = True
        N = len(self.data_source)
        count = 0
        for _ in range(N):
            if direction:
                yield count
                count += 1
            else:
                yield N-count
            direction = not direction

    def __len__(self):
        return len(self.data_source)

class IndicedSampler(Sampler):
    """
    A sampler that returns elements in a first-last order.
    """

    def __init__(self, data_source: Sized, indices):
        """
        :param data_source: Source of data, can be anything that has a len(),
        since we only care about its number of elements.
        """
        super().__init__(data_source)
        self.data_source = data_source
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)

def create_train_validation_loaders(
    dataset: Dataset, validation_ratio, batch_size=100, num_workers=2
):
    """
    Splits a dataset into a train and validation set, returning a
    DataLoader for each.
    :param dataset: The dataset to split.
    :param validation_ratio: Ratio (in range 0,1) of the validation set size to
        total dataset size.
    :param batch_size: Batch size the loaders will return from each set.
    :param num_workers: Number of workers to pass to dataloader init.
    :return: A tuple of train and validation DataLoader instances.
    """
    if not (0.0 < validation_ratio < 1.0):
        raise ValueError(validation_ratio)

    N = len(dataset)
    N_valid = math.floor(N * validation_ratio)
    N_train = N - N_valid

    train_indices = torch.arange(N_train)
    valid_indices = torch.arange(N_train, N)

    dl_train = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        sampler=IndicedSampler(dataset, train_indices)
        )
    dl_valid = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        sampler=IndicedSampler(dataset, valid_indices)
        )

    return dl_train, dl_valid
