from typing import Union, List
import numpy as np
from collections import Counter
from fedland.loaders import PartitionedDataLoader, load_mnist_data


def calc_gini(x: Union[np.ndarray, List[float]]) -> float:
    x = np.asarray(x)
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))

    return diffsum / (len(x)**2 * np.mean(x))


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

    LEFTOFF HERE
    print(f"Numpy version:")
    print(train_loader.dataset.data.numpy().shape)
    train_gini = calc_gini(Counter(trainset.targets.numpy()))
    print(f"Train gini: {train_gini}")
    test_gini = calc_gini(Counter(testset.targets.numpy()))
    print(f"Train gini: {train_gini}")
    # print(train_loader.dataset.data.numpy()[:1])
    # print(train_loader.dataset.targets.numpy()[:1])


if __name__ == "__main__":
    print("=>Running Scratch Test")
    test()
