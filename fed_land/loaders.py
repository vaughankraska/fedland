import os
import torchvision
from torch.utils.data import DataLoader


def load_mnist_data(batch_size=128, shuffle=True) -> tuple[DataLoader, DataLoader]:
    """
    Loads the MNIST Dataset using the built in torchvision data loaders.
    """
    out_dir = "./data/centralized"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # transform = torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize((0.1307,), (0.3081,))
    #     ])

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
            train_set, batch_size=batch_size, shuffle=shuffle
            )
    test_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=shuffle
            )

    return train_loader, test_loader
