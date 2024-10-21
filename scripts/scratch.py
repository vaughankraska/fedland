import numpy as np
from typing import Iterable
from collections import Counter
from fedland.loaders import PartitionedDataLoader, load_mnist_data
from torch.utils.data import Dataset, DataLoader


def gini(x: Iterable[float]):
    x = np.asarray(list(x))
    print(f'x: {x}')
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x)**2 * np.mean(x))


def calculate_class_balance(dataset: Dataset) -> dict:
    """
    Calculate class balance for a dataset. Assumes classes in target.

    Args:
        dataset (Dataset): A PyTorch Dataset object.
    Returns:
        dict: A dictionary containing
            'class_counts',
            'class_frequencies',
            'gini_index',
    """
    if not isinstance(dataset, Dataset):
        raise ValueError("Input must be a PyTorch Dataset object")
    all_labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        all_labels.append(label)

    class_counts = Counter(all_labels)

    total_samples = len(all_labels)

    class_frequencies = {cls: count / total_samples for cls, count in class_counts.items()}

    # Calculate Gini index (from econ, just curious if it summarizes imbalance)
    gini_index = gini(class_counts.values())
    print(f"GINI: {gini_index}")

    return {
        "class_counts": dict(class_counts),
        "class_frequencies": class_frequencies,
        "gini_index": gini_index,
    }


def test():
    trainset, testset = load_mnist_data()
    print(f"Train=\n{trainset}, Test=\n{testset}")

    train_loader = PartitionedDataLoader(
            trainset,
            num_partitions=1,
            partition_index=0,
            batch_size=1,
            shuffle=False
            )
    t_loader_0 = PartitionedDataLoader(
            trainset,
            num_partitions=2,
            partition_index=0,
            batch_size=1,
            shuffle=False
            )
    t_loader_1 = PartitionedDataLoader(
            trainset,
            num_partitions=2,
            partition_index=1,
            batch_size=1,
            shuffle=False
            )

    print(f"\nNumpy version:")
    print("\nFull:")
    print(train_loader.dataset.numpy().shape)
    print("\nP0:")
    print(t_loader_0.dataset.numpy().shape)
    print("\nP1:")
    print(t_loader_1.dataset.numpy().shape)

    print(f"\n0 indices: {t_loader_0.partition_indices}")
    print(f"len: {len(t_loader_0.partition_indices)}")

    print(f"\n1 indices: {t_loader_1.partition_indices}")
    print(f"len: {len(t_loader_1.partition_indices)}")
    for i, data in enumerate(t_loader_0):
        inputs = data[0]
        labels = data[1]
        if i % 1_000 == 999:
            print(f'in: {inputs.numpy().shape}, labels = {labels}')


if __name__ == "__main__":
    print("=>Running Scratch Test")
    test()
