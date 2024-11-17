import os
import json
from typing import Tuple
from fedn import APIClient
from fedland.loaders import PartitionedDataLoader, load_dataset
from fedland.database_models.experiment import Experiment
from fedland.metrics import calculate_class_balance

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)


def load_data(
    client_data_path, batch_size=128
) -> Tuple[PartitionedDataLoader, PartitionedDataLoader]:
    """Load data from disk.

    :param data_path: Path to data dir. ex) 'data/path/to'
    :type data_path: str
    :param batch_size: Batch Size for DataLoader
    :type batch_size: int
    :return: Tuple of Test and Training Loaders
    :rtype: Tuple[DataLoader, DataLoader]
    """
    if client_data_path is None:
        print("[*] Client Data path is None")
        client_data_path = os.environ.get("FEDN_DATA_PATH", abs_path + "/data/")

    experiment = get_experiment()

    # Get active clients from server
    training, testing = load_dataset(experiment.dataset_name)
    api_host = os.environ.get("FEDN_SERVER_HOST", "localhost")
    api_port = os.environ.get("FEDN_SERVER_PORT", 8092)
    api = APIClient(api_host, api_port)
    clients = api.get_active_clients()
    clients_count = clients.get("count")

    # The client's name should be set in the env
    client_id = os.environ.get("CLIENT_ID")
    client_index = next(
        (
            index
            for index, client in enumerate(clients["result"])
            if client["name"] == client_id
        ),
        None,
    )
    print(
        f"[*] Client {client_id} loading partition {client_index + 1}/{clients_count}"
    )
    assert (
        client_index is not None
    ), "Client name couldnt be found in the server's clients"

    # Set balance ratios and subset fractions if they exist
    target_balance_ratios = (
        experiment.target_balance_ratios[client_index]
        if experiment.target_balance_ratios
        and len(experiment.target_balance_ratios) > client_index
        else None
    )
    subset_fraction = (
        experiment.subset_fractions[client_index]
        if experiment.subset_fractions
        and len(experiment.subset_fractions) > client_index
        else 1
    )

    train_loader = PartitionedDataLoader(
        training,
        num_partitions=clients_count,
        partition_index=client_index,
        batch_size=batch_size,
        target_balance_ratios=target_balance_ratios,
        subset_fraction=subset_fraction,
        shuffle=False,
    )
    test_loader = PartitionedDataLoader(
        testing,
        num_partitions=clients_count,
        partition_index=client_index,
        batch_size=batch_size,
        target_balance_ratios=target_balance_ratios,
        subset_fraction=subset_fraction,
        shuffle=False,
    )

    client_info = {
        "experiment_id": experiment.id,
        "client_index": client_index,
        "data_indices": train_loader.partition_indices.tolist(),
        "balance": calculate_class_balance(train_loader),
        "local_rounds": [],
    }

    results_dir = os.environ.get("RESULTS_DIR")
    directory = f"{results_dir}/{experiment.id}/{client_id}"
    filename = f"{directory}/client.json"
    # Create all parent directories if they don't exist
    os.makedirs(directory, exist_ok=True)
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            json.dump(client_info, f)

    return train_loader, test_loader


def get_experiment():
    results_dir = os.environ.get("RESULTS_DIR")
    test_id = os.environ.get("TEST_ID")
    target_experiment = None
    with open(f"{results_dir}/experiments.json", "r") as f:
        tests = json.load(f)
        for t in tests:
            if t["id"] == test_id:
                target_experiment = Experiment(**t)
    return target_experiment


if __name__ == "__main__":
    print("[*] __main__ data.py")
    target_experiment = get_experiment()
    # Prepare data if not already done
    if target_experiment is not None:
        print(f"[*] Preloading dataset {target_experiment.dataset_name}")
        load_dataset(target_experiment.dataset_name)
