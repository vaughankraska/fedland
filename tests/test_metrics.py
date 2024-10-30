import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from fedland.metrics import calculate_class_balance
import numpy as np
from fedland.metrics.frobenius import (
    frobenius_norm,
    hessian_eigs,
    _npvec_to_tensorlist,
    _gradtensor_to_npvec,
    _eval_hess_vec_prod,
)


def test_class_balance(dataset: Dataset):
    loader = DataLoader(dataset=dataset)
    class_balances = calculate_class_balance(loader)

    assert class_balances.get("class_counts"), "class_counts key missing"
    assert class_balances.get("class_frequencies"), "class_frequencies key missing"
    assert class_balances.get("gini_index"), "gini_index key missing"
    gini = class_balances.get("gini_index")
    assert gini > 0.0 and gini < 1.0


# >>>begin AI generated test
def test_frobenius_norm_shape(model, dummy_data, criterion, device):
    norm = frobenius_norm(model, dummy_data, criterion, device, num_max=3)
    assert isinstance(norm, float)
    assert norm >= 0


def test_hessian_eigs_shape(model, dummy_data, criterion, device):
    eigenvalues, eigenvectors = hessian_eigs(
        model, dummy_data, criterion, device, num_max=3
    )

    # Check shapes
    num_params = sum(p.numel() for p in model.parameters() if len(p.shape) != 1)
    assert len(eigenvalues) == 3
    assert eigenvectors.shape == (num_params, 3)

    # Check eigenvalues are real
    assert np.all(np.isreal(eigenvalues))


def test_npvec_to_tensorlist(model, device):
    params = [p for p in model.parameters() if len(p.shape) != 1]
    total_params = sum(p.numel() for p in params)

    # Create dummy numpy vector
    vec = np.random.randn(total_params)

    # Convert to tensor list
    tensor_list = _npvec_to_tensorlist(vec, params, device)
    # Check properties

    assert len(tensor_list) == len(params)
    for t, p in zip(tensor_list, params):
        assert t.shape == p.shape
        assert t.device == device


def test_gradtensor_to_npvec(model):
    # Set some dummy gradients
    for p in model.parameters():
        p.grad = torch.randn_like(p)

    # Get numpy vector
    grad_vec = _gradtensor_to_npvec(model, include_bn=False)

    # Check shape
    expected_size = sum(p.numel() for p in model.parameters() if len(p.data.size()) > 1)
    assert grad_vec.shape == (expected_size,)


def test_eval_hess_vec_prod(model, dummy_data, criterion, device):
    params = [p for p in model.parameters() if len(p.shape) != 1]
    vec = [torch.randn_like(p) for p in params]

    # Initial gradients should be zero
    model.zero_grad()
    for p in model.parameters():
        assert torch.all(p.grad is None if p.grad is None else p.grad == 0)

    # Evaluate Hessian-vector product
    _eval_hess_vec_prod(vec, params, model, criterion, dummy_data, device)

    # Check that gradients are computed
    for p in params:
        assert p.grad is not None
        assert not torch.all(p.grad == 0)


def test_hessian_eigs_positive_semidefinite(model, dummy_data, criterion, device):
    # For MSE loss, Hessian should be positive semidefinite
    eigenvalues, _ = hessian_eigs(model, dummy_data, criterion, device, num_max=3)
    assert np.all(eigenvalues >= -1e-5)  # Allow for small numerical errors


@pytest.mark.parametrize("num_max", [1, 3, 5])
def test_hessian_eigs_num_max(model, dummy_data, criterion, device, num_max):
    eigenvalues, eigenvectors = hessian_eigs(
        model, dummy_data, criterion, device, num_max=num_max
    )
    assert len(eigenvalues) == num_max
    assert eigenvectors.shape[1] == num_max


def test_frobenius_norm_zero_for_constant_function(device):
    # Create a model that always outputs a constant
    class ConstantNet(nn.Module):
        def forward(self, x):
            return torch.ones(x.shape[0], 1)

    model = ConstantNet()
    X = torch.randn(10, 2)
    y = torch.randn(10, 1)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=5)
    criterion = nn.MSELoss()
    # Since the model has no parameters, its Hessian should be zero
    norm = frobenius_norm(model, dataloader, criterion, device, num_max=3)
    assert abs(norm) < 1e-5


def test_input_validation():
    with pytest.raises(Exception):
        # Test with invalid num_max
        _npvec_to_tensorlist(np.zeros(100), [], torch.device("cpu"))

    with pytest.raises(AssertionError):
        # Test with mismatched vector size
        params = [torch.randn(2, 3)]
        _npvec_to_tensorlist(np.zeros(10), params, torch.device("cpu"))


@pytest.mark.xfail(reason="Unimplemented")
def test_pac_bayes(dataset: Dataset):
    assert False, "TODO"


@pytest.mark.xfail(reason="Unimplemented")
def test_frobenius(dataset: Dataset):
    assert False, "TODO"


# end AI generated test>>>
