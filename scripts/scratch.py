import numpy as np
from fedland.loaders import PartitionedDataLoader, load_mnist_data
from fedland.metrics import calculate_class_balance


def test():
    trainset, testset = load_mnist_data()
    print(f"Train=\n{trainset}, Test=\n{testset}")

    train_loader = PartitionedDataLoader(
            trainset,
            num_partitions=1,
            partition_index=0,
            batch_size=1,
            )
    t_loader_0 = PartitionedDataLoader(
            trainset,
            num_partitions=2,
            partition_index=0,
            batch_size=1,
            )
    t_loader_1 = PartitionedDataLoader(
            trainset,
            num_partitions=2,
            partition_index=1,
            batch_size=1,
            )
    weighted_loader = PartitionedDataLoader(
            trainset,
            num_partitions=2,
            partition_index=0,
            batch_size=1,
            target_balance_ratios=list(np.arange(0, 10, step=0.1))
            )

    print("\nFull:")
    print(train_loader.dataset.data.numpy().shape)
    print("\nP0:")
    print(len(t_loader_0))
    print("\nP1:")
    print(len(t_loader_1))

    print(f"\n0 indices: {t_loader_0.partition_indices}")
    print(f"len: {len(t_loader_0.partition_indices)}")

    print(f"\n1 indices: {t_loader_1.partition_indices}")
    print(f"len: {len(t_loader_1.partition_indices)}")

    # for i, data in enumerate(t_loader_0):
    #     inputs = data[0]
    #     labels = data[1]
    #     if i % 1_000 == 999:
    #         print(f'{i} in: {inputs.numpy().shape}, labels = {labels}')

    print(f"zero set length {len(set(t_loader_0.partition_indices))}")
    print(f"first set length {len(set(t_loader_1.partition_indices))}")
    print(f"combined set length {len(set([*t_loader_0.partition_indices, *t_loader_1.partition_indices]))}")

    print(calculate_class_balance(dataloader=weighted_loader))


if __name__ == "__main__":
    print("=>Running Scratch Test")
    test()
