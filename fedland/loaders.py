import os
import torch
import torchvision
import numpy as np
from typing import List, Sequence, Iterator
from enum import Enum
from torch.utils.data import (
    DataLoader,
    Dataset,
    Sampler,
    SubsetRandomSampler,
)

# TODO! getting a little beefy, split into other files/module

OUT_DIR = "./data"
FIXED_SEED = 42


class DatasetIdentifier(Enum):
    CIFAR = "CIFAR"
    CIFAR100 = "CIFAR-100"
    MNIST = "MNIST"


class SubsetWeightedRandomSampler(Sampler[int]):
    """Samples elements from a given list of indices with specified weights,
    with replacement - Hybrid between SubsetSampler and WeightedRandomSampler
    from Pytorch.

    Args:
        weights (Tensor): a sequence of weights, not necessary summing up to
        one num_samples (int): number of samples to draw
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """

    def __init__(
        self,
        weights: torch.Tensor,
        num_samples: int,
        indices: Sequence[int],
        generator=None,
    ) -> None:
        self.weights = weights
        self.num_samples = num_samples
        self.indices = indices
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        rand_tensor = torch.multinomial(
            self.weights, self.num_samples, replacement=True, generator=self.generator
        )
        for idx in rand_tensor:
            yield self.indices[idx]

    def __len__(self) -> int:
        return self.num_samples


class PartitionedDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        num_partitions: int,
        partition_index: int,
        batch_size=128,
        target_balance_ratios: List[float] = None,
        subset_fraction=1,
        *args,
        **kwargs,
    ):
        if partition_index >= num_partitions:
            raise ValueError("partition_index cannot be greater than num_partitions")

        if num_partitions <= 0:
            raise ValueError("num_partitions must be non-zero and postive")

        labels = np.array([dataset[i][1] for i in range(len(dataset))])
        uni_labels = np.unique(labels)
        if target_balance_ratios is not None and len(target_balance_ratios) != len(
            uni_labels
        ):
            raise ValueError(
                f"target_balance_ratios (len {len(target_balance_ratios)})"
                f" must match dataset labels len({len(uni_labels)})"
            )

        if subset_fraction > 1 or subset_fraction < 0:
            raise ValueError(f"subset_fraction={subset_fraction} must be > 0 and <= 1")

        self.num_partitions = num_partitions
        self.target_balance_ratios = target_balance_ratios

        # Fix rng seed since we want reproducibility in data partitions
        rng = np.random.default_rng(FIXED_SEED)
        indices = np.arange(len(dataset))
        rng.shuffle(indices)

        # Subset the indices
        partition_size = len(dataset) // num_partitions
        start_idx = partition_index * partition_size
        end_idx = start_idx + partition_size
        self.partition_indices = rng.choice(
            indices[start_idx:end_idx],
            int(partition_size * subset_fraction),
            replace=False,
        )
        partition_labels = labels[self.partition_indices]

        if target_balance_ratios is None:
            sampler = SubsetRandomSampler(indices=self.partition_indices)
        else:
            # Sample the specified target ratios (assumes classes)
            current_class_counts = np.bincount(
                partition_labels, minlength=len(uni_labels)
            )
            current_class_ratios = current_class_counts / len(partition_labels)

            # Calculate weights for each class
            weights = np.zeros(len(partition_labels))
            for label_idx, target_ratio in enumerate(target_balance_ratios):
                if current_class_ratios[label_idx] > 0:
                    label_weight = target_ratio / current_class_ratios[label_idx]
                    weights[partition_labels == label_idx] = label_weight

            weights = torch.from_numpy(weights).float()
            sampler = SubsetWeightedRandomSampler(
                indices=self.partition_indices,
                weights=weights,
                num_samples=partition_size,
            )

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            *args,
            **kwargs,
        )


def load_cifar_data() -> tuple[Dataset, Dataset]:
    """
    Loads the CIFAR10 Dataset using the built in torchvision data loaders.

    returns:
        Tuple[Dataset, Dataset]: Tuple of training and testing Datasets
    """
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    train_set = torchvision.datasets.CIFAR10(
        root=f"{OUT_DIR}/train",
        transform=torchvision.transforms.ToTensor(),
        train=True,
        download=True,
    )
    test_set = torchvision.datasets.CIFAR10(
        root=f"{OUT_DIR}/test",
        transform=torchvision.transforms.ToTensor(),
        train=False,
        download=True,
    )

    return train_set, test_set


def load_cifar_100_data() -> tuple[Dataset, Dataset]:
    """
    Loads the CIFAR10 Dataset using the built in torchvision data loaders.

    returns:
        Tuple[Dataset, Dataset]: Tuple of training and testing Datasets
    """
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    train_set = torchvision.datasets.CIFAR100(
        root=f"{OUT_DIR}/train",
        transform=torchvision.transforms.ToTensor(),
        train=True,
        download=True,
    )
    test_set = torchvision.datasets.CIFAR100(
        root=f"{OUT_DIR}/test",
        transform=torchvision.transforms.ToTensor(),
        train=False,
        download=True,
    )

    return train_set, test_set


def load_mnist_data() -> tuple[Dataset, Dataset]:
    """
    Loads the MNIST Dataset using the built in torchvision data loaders.

    returns:
        Tuple[Dataset, Dataset]: Tuple of training and testing Datasets
    """
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


DATA_LOADER_MAP = {
    "CIFAR": load_cifar_data,
    "CIFAR-100": load_cifar_100_data,
    "MNIST": load_mnist_data,
}


def load_dataset(identifier: DatasetIdentifier) -> tuple[Dataset, Dataset]:
    if identifier in DATA_LOADER_MAP:
        return DATA_LOADER_MAP[identifier]()
    else:
        raise ValueError(f"Unknown dataset identifier: {identifier}")
