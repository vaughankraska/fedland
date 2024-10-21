import os
import socket
from typing import Tuple
from fedn import APIClient
from fedland.loaders import load_mnist_data, PartitionedDataLoader
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

    training, testing = load_mnist_data()
    # TODO: figure out the how to partition from centralized call
    api_host = os.environ.get("FEDN_SERVER_HOST", "api-server")
    api_port = os.environ.get("FEDN_SERVER_PORT", 8092)
    api = APIClient(api_host, api_port)
    clients = api.get_clients()
    clients_count = clients.get("count", 0)

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
        f"[*] Client {this_clients_name} loading partition {client_index}/{clients_count}"
    )
    assert (
        client_index is not None
    ), "Client name couldnt be found in the server's clients"

    train_loader = PartitionedDataLoader(
        training,
        num_partitions=clients_count,
        partition_index=client_index,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = PartitionedDataLoader(
        testing,
        num_partitions=clients_count,
        partition_index=client_index,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader


if __name__ == "__main__":
    # Prepare data if not already done
    if not os.path.exists(abs_path + "/data"):
        load_mnist_data()
