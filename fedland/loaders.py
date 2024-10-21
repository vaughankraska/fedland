import os
import torch
import torchvision
import numpy as np
from typing import List
from torch.utils.data import DataLoader, Dataset, Subset
OUT_DIR = "./data"
TORCH_SEED = 0
NUMPY_SEED = 42


# TODO: test it
# TODO: write balanced/imbalanced, IID non-IID loaders
class PartitionedDataLoader(DataLoader):
    def __init__(
            self,
            dataset: Dataset,
            num_partitions: int,
            partition_index: int,
            batch_size=128,
            shuffle=False,
            target_balance_ratios: List[float] = None,
            *args, **kwargs
            ):
        self.num_paritions = num_partitions
        self.shuffle = shuffle
        self.target_balance_ratios = target_balance_ratios

        # Fix rng seed since we want reproducibility.
        rng = np.random.default_rng(NUMPY_SEED)
        indices = np.arange(len(dataset))
        rng.shuffle(indices)

        # Subset the data
        partition_size = len(dataset) // num_partitions
        start_idx = partition_index * partition_size
        end_idx = start_idx + partition_size
        self.partition_indices = indices[start_idx:end_idx]
        if shuffle:
            indices = rng.permutation(self.partition_indices)
        else:
            indices = self.partition_indices
        dataset = dataset.__getitems__([indices[idx] for idx in indices])

        super().__init__(dataset=dataset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         *args, **kwargs)


def load_mnist_data() -> tuple[Dataset, Dataset]:
    """
    Loads the MNIST Dataset using the built in torchvision data loaders.

    returns:
        Tuple[Dataset, Dataset]: Tuple of training and testing Datasets
    """
    torch.manual_seed(TORCH_SEED)
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    train_set = torchvision.datasets.MNIST(
            root=f"{OUT_DIR}/train",
            transform=torchvision.transforms.ToTensor(),
            train=True,
            download=True,
            )
    test_set = torchvision.datasets.MNIST(
            root=f"{OUT_DIR}/test",
            transform=torchvision.transforms.ToTensor(),
            train=False,
            download=True,
            )

    return train_set, test_set
