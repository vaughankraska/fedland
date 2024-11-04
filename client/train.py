import os
import sys
import socket
import torch
import torch.optim as optim
from datetime import datetime
from fedn import APIClient
from fedn.utils.helpers.helpers import save_metadata
from fedn.network.api.v1.session_routes import session_store
from fedland.metrics import path_norm
from fedland.database_models.experiment import experiment_store
from model import load_parameters, save_parameters
from data import load_data

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))


def train(
    in_model_path,
    out_model_path,
    data_path=None,
    batch_size=64,
    epochs=5,
    lr=0.01,
    momentum=0.5,
):
    """Complete a model update.

    Load model paramters from in_model_path (managed by the FEDn client),
    perform a model update, and write updated paramters
    to out_model_path (picked up by the FEDn client).

    Defaults match the centralized baseline example and (should) match
    the methods done by Horvath.

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_model_path: The path to save the output model to.
    :type out_model_path: str
    :param data_path: The path to the data file.
    :type data_path: str
    :param batch_size: The batch size to use.
    :type batch_size: int
    :param epochs: The number of epochs to train.
    :type epochs: int
    :param lr: The learning rate to use.
    :type lr: float
    :param momentum: The momentum rate to use.
    :type momentum: float
    """
    # TODO: cleanup or find something better
    api_host = os.environ.get("FEDN_SERVER_HOST", "api-server")
    api_port = os.environ.get("FEDN_SERVER_PORT", 8092)
    api = APIClient(api_host, api_port)
    clients = api.get_clients()
    this_clients_name = socket.gethostname()
    client_index = next(
        (
            index
            for index, client in enumerate(clients["result"])
            if client["name"] == this_clients_name
        ),
        None,
    )

    session = session_store.list(limit=1, skip=0, sort_key="session_id")
    session_id = session.get("result")[0]["session_id"]
    experiment = experiment_store.get_latest()
    experiment_id = experiment.id if experiment else None

    # Load data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_data(data_path, batch_size=batch_size)

    # Load parmeters and initialize model
    model = load_parameters(in_model_path).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    print(f"[*] Model: {model}\n")
    print(f"[*] Device: {device}")
    print(f"[*] Criterion: {criterion}")
    print(f"[*] Epochs: {epochs}")
    print(f"[*] Batch Size: {batch_size}")
    print(f"[*] session_id: {session_id}")
    print(f"[*] client_index: {client_index}")

    # Train
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for i, data in enumerate(train_loader):
            inputs = data[0].to(device)
            labels = data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Do reporting each local epoch
        train_accuracy = 100.0 * correct / total
        train_loss = running_loss / len(train_loader)
        model.eval()
        test_running_loss, test_correct, test_total = 0.0, 0, 0
        with torch.no_grad():
            for test_data in test_loader:
                test_inputs = test_data[0].to(device)
                test_labels = test_data[1].to(device)
                test_outputs = model(test_inputs)
                test_loss = criterion(test_outputs, test_labels)
                test_running_loss += test_loss.item()
                _, predicted = test_outputs.max(1)
                test_total += test_labels.size(0)
                test_correct += predicted.eq(test_labels).sum().item()
        # Calculate test metrics
        test_accuracy = 100.0 * test_correct / test_total
        test_loss = test_running_loss / len(test_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}],\n"  # noqa E501
              f"Train Loss: {running_loss:.3f}, Train Accuracy: {train_accuracy:.2f}%,\n"  # noqa E501
              f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.2f}%\n"  # noqa E501
              )
        try:
            pn = path_norm(model, train_loader)
            print(f"PathNorm: {pn:.4f}\n")  # noqa E501
            new_local_round = {
                "session_id": session_id,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "path_norm": pn,
                "pac_bayes_bound": 0.0,  # no implementing but have code expecting it
                "frobenius_norm": 0.0,
                "timestamp": datetime.now().isoformat(),
            }
            # TODO: fix dump to mongo
            did_insert = experiment_store.client_stat_store.append_local_round(
                experiment_id=experiment_id,
                client_index=client_index,
                local_round=new_local_round,
            )
            print(f"[*] Appended Stats? {did_insert} for {experiment_id}")
        except Exception as e:
            print(f"[!!!] Error in stats: {e}")

    # Metadata needed for aggregation server side
    metadata = {
        # num_examples are mandatory
        "num_examples": len(train_loader.dataset),
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
    }

    # Save JSON metadata file (mandatory)
    save_metadata(metadata, out_model_path)

    # Save model update (mandatory)
    save_parameters(model, out_model_path)


if __name__ == "__main__":
    print(f"[*] SYS ARGS: {sys.argv}")
    train(sys.argv[1], sys.argv[2])
