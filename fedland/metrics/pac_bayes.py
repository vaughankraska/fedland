import copy
from typing import Optional, Tuple
import torch
from torch.nn import Module
from torch.utils.data import DataLoader


def modify_model(model: Module, mu: float) -> Module:
    modified_model = copy.deepcopy(model)
    index = 0
    with torch.no_grad():
        for name, param in modified_model.named_parameters():
            param.data += torch.normal(torch.zeros_like(param), mu)
            index += param.numel()

    return modified_model


def estimate_accuracy(
    model: Module,
    data_loader: DataLoader,
    criterion: Module,
    device: torch.device,
    M3: Optional[int] = None,
) -> float:
    loss = 0
    total = 0
    if M3 is None:
        model.eval()
        with torch.no_grad():
            for data in data_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss += criterion(outputs, labels).item()
                total += labels.size(0)
    else:
        model.eval()
        with torch.no_grad():
            for _ in range(M3):
                inputs, labels = next(iter(data_loader))
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss += criterion(outputs, labels).item()
                total += labels.size(0)

    return loss / total


def eval_modified_model(
    model: Module,
    data_loader: DataLoader,
    criterion: Module,
    device: torch.device,
    sigma_new: float,
    M3: int,
) -> float:
    modified_model = modify_model(model, sigma_new**2)
    acc = estimate_accuracy(
        model=modified_model,
        data_loader=data_loader,
        criterion=criterion,
        device=device,
        M3=M3,
    )

    return acc / data_loader.batch_size


def pac_bayes_bound(
    model: Module,
    data_loader: DataLoader,
    criterion: Module,
    device: torch.device,
    d: float,
    sigma_max: float = 1,
    sigma_min: float = 0,
    M1: int = 100,
    M2: int = 100,
    M3: int = 100,
) -> Tuple[float, float]:
    """
    Calculates the Pac Bayes Bound for a NN.
    "[Pac Bayes] with d = 0.1, σ max = 0.5, σ min = 10−2,
    M1 = 30, M2 = 10, M3 = 20, ϵd = 10−4 for the ResNet18"
    """
    model_accuracy = estimate_accuracy(
        model=model,
        data_loader=data_loader,
        criterion=criterion,
        device=device,
        M3=None,
    )
    for i in range(M1):
        sigma_new = (sigma_max + sigma_min) / 2
        acc = 0
        for j in range(M2):
            acc += eval_modified_model(
                model=model,
                data_loader=data_loader,
                criterion=criterion,
                device=device,
                sigma_new=sigma_new,
                M3=M3,
            )
        est_acc = acc / M2
        dev = abs(model_accuracy - est_acc)
        if dev < 1e-4 or sigma_max - sigma_min < 1e-4:
            return sigma_new
        if dev > d:
            sigma_max = sigma_new
        else:
            sigma_min = sigma_new

    return sigma_min, sigma_max
