import os
import torchvision
from torch.utils.data import DataLoader


def load_mnist_data(
        batch_size=128,
        shuffle_train=True,
        shuffle_test=True
        ) -> tuple[DataLoader, DataLoader]:
    """
    Loads the MNIST Dataset using the built in torchvision data loaders.
    """
    out_dir = "./data"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    train_set = torchvision.datasets.MNIST(
            root=f"{out_dir}/train",
            transform=torchvision.transforms.ToTensor(),
            train=True,
            download=True,
            )
    test_set = torchvision.datasets.MNIST(
            root=f"{out_dir}/test",
            transform=torchvision.transforms.ToTensor(),
            train=False,
            download=True,
            )

    train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=shuffle_train
            )
    test_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=shuffle_test
            )

    return train_loader, test_loader
