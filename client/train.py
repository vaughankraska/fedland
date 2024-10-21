import os
import sys
import torch
import torch.optim as optim
from fedn.utils.helpers.helpers import save_metadata
from fedland.metrics import path_norm
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
    # Load data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _ = load_data(data_path, batch_size=batch_size)

    # Load parmeters and initialize model
    model = load_parameters(in_model_path).to(device)
    print(f"[*] Model: {model}")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

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

            if i % 200 == 199:
                p_norm = path_norm(model, train_loader)
                print(
                    f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], "  # noqa E501
                    f"Loss: {running_loss/100:.3f}, Accuracy: {100.*correct/total:.2f}%"  # noqa E501
                    f"PNorm: {p_norm:.4f}"
                )
                running_loss, correct, total = 0.0, 0, 0

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
