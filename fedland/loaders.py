import os
import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader, Dataset
OUT_DIR = "./data"
TORCH_SEED = 0
NUMPY_SEED = 42


# TODO: test it
# TODO: write balanced/imbalanced, IID non-IID loaders
class PartitionedDataLoader(DataLoader):
    def __init__(
            self,
            dataset,
            num_partitions,
            partition_index,
            batch_size=128,
            shuffle=False,
            drop_last=False,
            *args, **kwargs
            ):
        self.dataset = dataset
        self.num_paritions = num_partitions
        self.shuffle = shuffle
        self.drop_last = drop_last

        rng = np.random.default_rng(NUMPY_SEED)
        indices = np.arange(len(dataset))
        rng.shuffle(indices)

        partition_size = len(dataset) // num_partitions
        start_idx = partition_index * partition_size
        end_idx = start_idx + partition_size
        self.partition_indices = indices[start_idx:end_idx]

        super().__init__(dataset=dataset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         drop_last=drop_last,
                         *args, **kwargs)

    def __iter__(self):
        if self.shuffle:
            rng = np.random.default_rng(NUMPY_SEED)
            indices = rng.permutation(self.partition_indices)
        else:
            indices = self.partition_indices

        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        if not self.drop_last and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]

        for batch_indices in batches:
            yield [self.dataset[i] for i in batch_indices]

    def __len__(self):
        if self.drop_last:
            return len(self.partition_indices) // self.batch_size
        else:
            return (len(self.partition_indices) + self.batch_size - 1) // self.batch_size


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
