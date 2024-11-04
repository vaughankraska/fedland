import os
import socket
import time
from typing import Tuple
from fedn import APIClient
from fedland.loaders import PartitionedDataLoader, load_dataset
from fedland.metrics import calculate_class_balance
from fedland.database_models.client_stat import ClientStat
from fedland.database_models.experiment import experiment_store
from torch.utils.data import DataLoader

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)


def load_data(client_data_path, batch_size=128) -> Tuple[DataLoader, DataLoader]:
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

    latest_experiment = experiment_store.get_latest()
    while latest_experiment is None:
        print("[*] Waiting for Experiment...")
        time.sleep(5)
        latest_experiment = experiment_store.get_latest()

    training, testing = load_dataset(latest_experiment.dataset_name)
    api_host = os.environ.get("FEDN_SERVER_HOST", "api-server")
    api_port = os.environ.get("FEDN_SERVER_PORT", 8092)
    api = APIClient(api_host, api_port)
    clients = api.get_active_clients()
    clients_count = clients.get("count")

    # The client's name should be set to the hostname (which is the hash
    # from the docker container).
    this_clients_name = socket.gethostname()
    client_index = next(
        (
            index
            for index, client in enumerate(clients["result"])
            if client["name"] == this_clients_name
        ),
        None,
    )
    print(
        f"[*] Client {this_clients_name} loading partition {client_index + 1}/{clients_count}"
    )
    assert (
        client_index is not None
    ), "Client name couldnt be found in the server's clients"

    # Set balance ratios and subset fractions if they exist
    target_balance_ratios = (
        latest_experiment.target_balance_ratios[client_index]
        if latest_experiment.target_balance_ratios
        and len(latest_experiment.target_balance_ratios) > client_index
        else None
    )
    subset_fraction = (
        latest_experiment.subset_fractions[client_index]
        if latest_experiment.subset_fractions
        and len(latest_experiment.subset_fractions) > client_index
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

    # Dont overwrite local rounds if they exist
    existing_stats = experiment_store.client_stat_store.get(
        experiment_id=latest_experiment.id, client_index=client_index, use_typing=True
    )
    client_stat = ClientStat(
        experiment_id=latest_experiment.id,
        client_index=client_index,
        data_indices=train_loader.partition_indices,
        balance=calculate_class_balance(train_loader),
        local_rounds=existing_stats.local_rounds if existing_stats else [],
    )
    succ = experiment_store.client_stat_store.create_or_update(client_stat)
    print(f"[*] ClientStat Added or Updated? {succ}")

    return train_loader, test_loader


if __name__ == "__main__":
    print("[*] __main__ data.py")
    # Prepare data if not already done (assume latest experiment dataset)
    latest_experiment = experiment_store.get_latest()
    if latest_experiment is not None:
        print(f"[*] Preloading dataset {latest_experiment.dataset_name}")
        load_dataset(latest_experiment.dataset_name)
