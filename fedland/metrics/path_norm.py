import torch
import copy
from torch.nn import Module
from torch.utils.data import DataLoader


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
            param.data = param.data**2

    ones = torch.ones_like(in_tensor)
    summed = torch.sum(modified_model.forward(ones)).data
    assert summed > 0, "Cannot square root a negative number in path"

    return summed**0.5
