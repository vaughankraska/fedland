# All based on/replicated from Horvath's Implementation
# MSC code, adapted for MNIST and a FEDn
from typing import Tuple
import copy
from fedland.metrics.frobenius import frobenius_norm
from fedland.metrics.pac_bayes import pac_bayes_bound
import torch
from torch.nn import Module
from torch.utils.data import DataLoader


def evaluate_all(
        model: Module,
        data_loader: DataLoader,
        criterion: Module,
        device: torch.device
        ) -> Tuple[float, float, float, Tuple[float, float], float]:
    """
    Evaluate all metrics:
        Avg Loss, Accuracy, Path Norm, Pac Bayes, Frobenius
    returns:
        Tuple[float, float, float, Tuple[float, float], float]
    """

    avg_loss, accuracy = evaluate(
            model,
            data_loader,
            criterion,
            device
            )

    in_size, _ = next(iter(data_loader))
    p_norm = path_norm(model, data_loader)

    bayes = pac_bayes_bound(
            model=model,
            data_loader=data_loader,
            criterion=criterion,
            device=device,
            d=0.1,
            sigma_max=0.5,
            sigma_min=0.001,
            M1=30,
            M2=10,
            M3=20
            )

    frobenius = frobenius_norm(
            model=model,
            data_loader=data_loader,
            criterion=criterion,
            device=device,
            num_max=10
            )

    return avg_loss, accuracy, p_norm, bayes, frobenius


def evaluate(
        model: Module,
        data_loader: DataLoader,
        criterion: Module,
        device: torch.device
        ) -> Tuple[float, float]:
    """
    Evaluate loss and accuracy metrics:
        Avg Loss, Accuracy
    returns:
        Tuple[float, float]
    """

    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / len(data_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def path_norm(model: Module, data_loader: DataLoader) -> float:
    """
    Calculate the Path Norm for a NN via forward pass.
    """
    modified_model = copy.deepcopy(model)
    modified_model.to(torch.device("cpu"))
    in_size, _ = next(iter(data_loader))
    in_tensor = in_size[0].unsqueeze(0)

    with torch.no_grad():
        for param in modified_model.parameters():
            param.data = param.data ** 2

    ones = torch.ones_like(in_tensor)
    summed = (torch.sum(modified_model.forward(ones)).data)
    assert summed > 0, "Cannot square root a negative number in path"

    return summed ** 0.5


