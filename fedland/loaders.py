import os
import torch
import torchvision
import numpy as np
from typing import List
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, WeightedRandomSampler
OUT_DIR = "./data"
FIXED_SEED = 42


# TODO: test it
# TODO: write balanced/imbalanced, IID non-IID loaders
class PartitionedDataLoader(DataLoader):
    def __init__(
            self,
            dataset: Dataset,
            num_partitions: int,
            partition_index: int,
            batch_size=128,
            target_balance_ratios: List[float] = None,
            *args, **kwargs
            ):

        if partition_index > num_partitions:
            raise ValueError("partition_index cannot be greater than num_partitions")
        if num_partitions <= 0:
            raise ValueError("num_partitions must be non-zero and postive")
        # TODO assert target_balance_ratios match dimensions of labels

        # Defaults for consistency
        generator = torch.Generator().manual_seed(FIXED_SEED)

        self.num_partitions = num_partitions
        self.target_balance_ratios = target_balance_ratios

        # Fix rng seed since we want reproducibility.
        rng = np.random.default_rng(FIXED_SEED)
        indices = np.arange(len(dataset))
        rng.shuffle(indices)

        # Subset the indices
        partition_size = len(dataset) // num_partitions
        start_idx = partition_index * partition_size
        end_idx = start_idx + partition_size
        self.partition_indices = indices[start_idx:end_idx]

        if target_balance_ratios is None:
            sampler = SubsetRandomSampler(
                    indices=self.partition_indices,
                    generator=generator
                    )
        else:
            sampler = WeightedRandomSampler(
                    weights=target_balance_ratios,
                    num_samples=partition_size,
                    generator=generator,
                    replacement=True,
                    )

        super().__init__(dataset=dataset,
                         batch_size=batch_size,
                         generator=generator,
                         sampler=sampler,
                         *args, **kwargs)


def load_mnist_data() -> tuple[Dataset, Dataset]:
    """
    Loads the MNIST Dataset using the built in torchvision data loaders.

    returns:
        Tuple[Dataset, Dataset]: Tuple of training and testing Datasets
    """
    torch.manual_seed(FIXED_SEED)
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
