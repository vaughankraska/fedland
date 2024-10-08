import os
import sys

import torch
from fedn.utils.helpers.helpers import save_metrics
from model import load_parameters

from data import load_data
from fedland.metrics import evaluate, path_norm

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))


def validate(in_model_path, out_json_path, data_path=None):
    """Validate model.

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_json_path: The path to save the output JSON to.
    :type out_json_path: str
    :param data_path: The path to the data file.
    :type data_path: str
    """
    # Load data
    train_loader, test_loader = load_data(data_path)

    # Load model
    model = load_parameters(in_model_path)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cpu")

    train_loss, train_acc = evaluate(model, train_loader, criterion, device)  # noqa E501
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    pnorm = path_norm(model, train_loader)
    # JSON schema
    report = {
        "training_loss": float(train_loss),
        "training_accuracy": float(train_acc),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "path_norm": float(pnorm)
    }

    # Save JSON
    save_metrics(report, out_json_path)


if __name__ == "__main__":
    validate(sys.argv[1], sys.argv[2])
