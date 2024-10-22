import pytest
import numpy as np
from collections import Counter
from fedland.loaders import PartitionedDataLoader


def test_partitioned_loader_fails_negative_params(dataset):
    with pytest.raises(ValueError):
        PartitionedDataLoader(dataset, 0, 0)


def test_partitioned_loader_fails_mismatched_params(dataset):
    with pytest.raises(ValueError):
        PartitionedDataLoader(dataset, 5, 5)


def test_partitioned_loader_fails_empty_weights(dataset):
    with pytest.raises(ValueError):
        PartitionedDataLoader(dataset, 2, 0, target_balance_ratios=[])


def test_partitioned_loader_fails_mismatched_weights(dataset):
    with pytest.raises(ValueError):
        PartitionedDataLoader(
                dataset, 2, 0, target_balance_ratios=[.1, .2, .3, .4]
                )


def test_partitioned_loader_with_target_ratios(dataset):
    target_ratios = [0.1, 0.6, 0.3]
    loader = PartitionedDataLoader(
            dataset, 3, 0, target_balance_ratios=target_ratios
            )

    sampled_labels = []
    for _, data in enumerate(loader):
        label = data[1]
        sampled_labels.extend(label.numpy())

    label_counts = Counter(sampled_labels)
    total_samples = len(sampled_labels)

    actual = {label: count / total_samples
              for label, count in label_counts.items()}
    expected = {label: ratio
                for label, ratio in zip(range(3), target_ratios)}

    for label in range(len(target_ratios)):
        expected_ratio = expected.get(label, 0)
        actual_ratio = actual.get(label, 0)

        assert abs(actual_ratio - expected_ratio) <= 0.05, (
                f"Label {label}: expected ratio {expected_ratio}, "
                f"actual ratio {actual_ratio}"
                )


def test_single_partition(dataset):
    num_partitions = 1

    loader = PartitionedDataLoader(
        dataset,
        num_partitions=num_partitions,
        partition_index=0,
        batch_size=1,
    )
    for _, data in enumerate(loader):
        _ = data[0]
        _ = data[1]

    assert len(loader) == len(dataset)


def test_two_partitions(dataset):
    num_partitions = 2

    loader_0 = PartitionedDataLoader(
        dataset,
        num_partitions=num_partitions,
        partition_index=0,
        batch_size=1,
    )
    loader_1 = PartitionedDataLoader(
        dataset,
        num_partitions=num_partitions,
        partition_index=1,
        batch_size=1,
    )
    indices_set = set([
        *loader_0.partition_indices,
        *loader_1.partition_indices
        ])

    assert len(loader_0) == len(loader_1), (
            "Loaders differ in length"
            )
    assert len(indices_set) == len(dataset), (
            "Indices bleeding across partitions"
            )
