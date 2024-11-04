import collections

import torch

from fedn.utils.helpers.helpers import get_helper
from fedland.networks import FedNet, CifarFedNet
from fedland.database_models.experiment import experiment_store

HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)


def compile_model(which_model):
    """Compile the pytorch model.

    :param which_model: The model type to compile.
    :type model: str
    :return: The compiled model.
    :rtype: torch.nn.Module
    """

    if which_model == "CifarFedNet":
        return CifarFedNet()
    elif which_model == "CifarFedNet-100":
        return CifarFedNet(num_classes=100)
    elif which_model == "Fednet":
        return FedNet()
    else:
        raise ValueError(f"Unknown Model Type '{which_model}', cannot compile")


def save_parameters(model, out_path):
    """Save model paramters to file.

    :param model: The model to serialize.
    :type model: torch.nn.Module
    :param out_path: The path to save to.
    :type out_path: str
    """
    parameters_np = [val.cpu().numpy() for _, val in model.state_dict().items()]
    helper.save(parameters_np, out_path)


def load_parameters(model_path) -> torch.nn.Module:
    """Load model parameters from file and populate model.

    :param model_path: The path to load from.
    :type model_path: str
    :return: The loaded model.
    :rtype: torch.nn.Module
    """
    latest_experiment = experiment_store.get_latest()
    model = compile_model(latest_experiment.model)
    parameters_np = helper.load(model_path)

    params_dict = zip(model.state_dict().keys(), parameters_np)
    state_dict = collections.OrderedDict(
        {key: torch.tensor(x) for key, x in params_dict}
    )
    model.load_state_dict(state_dict, strict=True)
    return model


def init_seed(out_path="seed.npz", which_model="FedNet"):
    """Initialize seed model and save it to file.

    :param out_path: The path to save the seed model to.
    :type out_path: str
    """
    # Init and save
    model = compile_model(which_model)
    save_parameters(model, out_path)


if __name__ == "__main__":
    print("[*] __main__ model.py")
    latest_experiment = experiment_store.get_latest()
    print(f"[*] Compiling model: {latest_experiment.model}")
    init_seed("../seed.npz", which_model=latest_experiment.model)
