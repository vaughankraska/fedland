import os
from typing import Tuple
from math import floor
from fedland.loaders import load_mnist_data
import torch
from torch.utils.data import Subset
from torch.utils.data import DataLoader

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)


def load_data(client_data_path, batch_size=128) -> Tuple[DataLoader, DataLoader]:
    """Load data from disk.

    :param data_path: Path to data dir. ex) 'data/clients/1'
    :type data_path: str
    :param batch_size: Batch Size for DataLoader
    :type batch_size: int
    :return: Tuple of Test and Training Loaders
    :rtype: Tuple[DataLoader, DataLoader]
    """
    if client_data_path is None:
        client_data_path = os.environ.get("FEDN_DATA_PATH", abs_path + "/data/clients/1")

    train_subset = torch.load(client_data_path + "train/mnist.data", weights_only=True)
    test_subset = torch.load(client_data_path + "test/mnist.data", weights_only=True)

    train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=False
            )
    test_loader = DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=False
            )

    return train_loader, test_loader


def split(out_dir="data"):
    n_splits = int(os.environ.get("FEDN_NUM_DATA_SPLITS", 5))

    # Make dir
    if not os.path.exists(f"{out_dir}/clients"):
        os.mkdir(f"{out_dir}/clients")

    # Load and convert to dict
    train_loader, test_loader = load_mnist_data(
            shuffle_train=True, shuffle_test=False
            )

    train_split_size = floor(len(train_loader.dataset) / n_splits)
    test_split_size = floor(len(test_loader.dataset) / n_splits)

    # Make splits
    for i in range(n_splits):
        subdir = f"{out_dir}/clients/{str(i+1)}"
        if not os.path.exists(subdir):
            os.mkdir(subdir)

        train_subset = Subset(
                train_loader.dataset,
                range(i * train_split_size, (i + 1) * train_split_size)
                )
        test_subset = Subset(
                test_loader.dataset,
                range(i * test_split_size, (i + 1) * test_split_size)
                )

        torch.save(train_subset, f"{subdir}/train/mnist.pt")
        torch.save(test_subset, f"{subdir}/test/mnist.pt")


if __name__ == "__main__":
    # Prepare data if not already done
    if not os.path.exists(abs_path + "/data/clients/1"):
        load_mnist_data()
        split()
