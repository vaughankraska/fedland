# All based on/replicated from Horvath's Implementation
# MSC code, adapted for MNIST and a FEDn
from typing import Tuple, Iterable, Any, Dict
from fedland.metrics.frobenius import frobenius_norm
from fedland.metrics.pac_bayes import pac_bayes_bound
from fedland.metrics.path_norm import path_norm
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from collections import Counter
import numpy as np


def evaluate_all(
    model: Module, data_loader: DataLoader, criterion: Module, device: torch.device
) -> Dict[str, Any]:
    """
    Evaluate all metrics:
        Avg Loss, Accuracy, Path Norm, Pac Bayes, Frobenius

    Args:
        model (Module): The model to evaluate
        data_loader (DataLoader): DataLoader for the dataset
        criterion (Module): Loss function
        device (torch.device): Device to run the evaluation on

    Returns:
        Dict[str, Any]: A dictionary containing all evaluation metrics
    """
    avg_loss, accuracy = evaluate(model, data_loader, criterion, device)

    in_size, _ = next(iter(data_loader))

    try:
        p_norm = path_norm(model, data_loader)
    except Exception as e:
        print(f"[!] Error calculating path norm: {e}")

    try:
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
            M3=20,
        )
    except Exception as e:
        print(f"[!] Error calculating pac bayes bound: {e}")

    try:
        frobenius = frobenius_norm(
            model=model,
            data_loader=data_loader,
            criterion=criterion,
            device=device,
            num_max=10,
        )
    except Exception as e:
        print(f"[!] Error Calculating frob norm: {e}")

    return {
        "avg_loss": avg_loss,
        "accuracy": accuracy,
        "path_norm": p_norm,
        "pac_bayes": bayes,
        "frobenius_norm": frobenius,
    }


def evaluate(
    model: Module, data_loader: DataLoader, criterion: Module, device: torch.device
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
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def calculate_class_balance(dataloader: DataLoader) -> Dict[str, Any]:
    """
    Calculate class balance for a dataset using a DataLoader.

    This will run for the entire dataset/subset as defined by the loader.
    No not use in training iterations since its wasteful and unnecessary.

    Args:
        dataloader (DataLoader): A PyTorch DataLoader object.
    Returns:
        dict: A dictionary containing
            'class_counts',
            'class_frequencies',
            'gini_index',
    """
    if not isinstance(dataloader, DataLoader):
        raise ValueError("Input must be a PyTorch DataLoader object")

    all_labels = []
    for batch in dataloader:
        _, labels = batch
        all_labels.extend(labels.tolist())

    class_counts = Counter(all_labels)
    total_samples = len(all_labels)
    class_frequencies = {
        f"class_{cls}": count / total_samples for cls, count in class_counts.items()
    }

    # Calculate Gini index
    gini_index = gini(class_counts.values())

    return {
        "class_counts": {f"class_{cls}": count for cls, count in class_counts.items()},
        "class_frequencies": class_frequencies,
        "gini_index": gini_index,
    }


def gini(x: Iterable[float]):
    x = np.asarray(list(x))
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))

    return diffsum / (len(x) ** 2 * np.mean(x))
