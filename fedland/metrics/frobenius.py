import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from scipy.sparse.linalg import LinearOperator, eigsh
import numpy as np


def frobenius_norm(
    model: Module,
    data_loader: DataLoader,
    criterion: Module,
    device: torch.device,
    num_max=10,
) -> float:
    """
    Calculate the Frobenius Norm of the Hessian estimated by
    the top num_max eigenvalues (instead of all).
    """
    w, _ = hessian_eigs(model, data_loader, criterion, device, num_max)
    return sum([e**2 for e in w]) ** 0.5


def hessian_eigs(
    model: Module,
    data_loader: DataLoader,
    criterion: Module,
    device: torch.device,
    num_max=10,
) -> tuple[list[float], list[list[float]]]:
    """
    Calculate the Hessian Eigenvalues from model parameters

    returns:
        - `w`: array
            Array of k eigenvalues.
        - `v`: array
            An array representing the `k` eigenvectors.  The column ``v[:, i]``
            is the eigenvector corresponding to the eigenvalue ``w[i]``.
    """
    params = [p for p in model.parameters() if len(p.shape) != 1]
    N = sum(p.numel() for p in params)

    def hess_vec_prod(vec):
        vec = _npvec_to_tensorlist(vec, params, device)
        _eval_hess_vec_prod(vec, params, model, criterion, data_loader, device)
        return _gradtensor_to_npvec(model)

    A = LinearOperator((N, N), matvec=hess_vec_prod)
    return eigsh(A, k=num_max, tol=1e-2)


def _npvec_to_tensorlist(vec, params, device):
    """
    Convert a numpy vector to a list of tensor with the same
    dimensions as params.

    Args:
       vec: a 1D numpy vector
       params: a list of parameters from net

    Returns:
       rval: a list of tensors with the same shape as params
    """
    loc = 0
    rval = []
    for p in params:
        numel = p.data.numel()
        rval.append(
            torch.from_numpy(vec[loc : loc + numel])
            .view(p.data.shape)
            .float()
            .to(device)
        )
        loc += numel

    assert loc == vec.size, "Vector has more elements than net has parameters"

    return rval


def _gradtensor_to_npvec(net, include_bn=False):
    """
    Extract gradients from net, and return a concatenated numpy vector.

    Args:
        net: trained model
        include_bn: If include_bn, then gradients w.r.t. BN parameters and bias
        values are also included.
        Otherwise only gradients with dim > 1 are considered.

    Returns:
        a concatenated numpy vector containing all gradients
    """

    # filter = lambda p: include_bn or len(p.data.size()) > 1
    def filter_function(p):
        return include_bn or len(p.data.size()) > 1

    return np.concatenate(
        [
            p.grad.data.cpu().numpy().ravel()
            for p in net.parameters()
            if filter_function(p)
        ]
    )


def _eval_hess_vec_prod(vec, params, net, criterion, data_loader, device):
    """
    Evaluate product of the Hessian of the loss function with a direction
    vector "vec". The product result is saved in the grad of net.
    Args:
        vec: a list of tensor with the same dimensions as "params".
        params: the parameter list of the net.
        net: model with trained parameters.
        criterion: loss function.
        data_loader: DataLoader for the dataset.
        device: cuda or cpu

    1. Perform usual pass through data_loader and compute the loss for
    each minibatch. Also, perform backward pass for each mini batch
    (NOT cleaning gradient as we will need the graient computed on the FULL dataset)
    Use grad_f = torch.autograd.grad(loss, inputs=params, create_graph=True)
    2. After that, loop parallelly through grad_f and vec and sum g * v.
    3. Finally, perform one more backward pass.
    """

    net.zero_grad()  # clears grad for every parameter in the net
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        output = net(x_batch)
        loss = criterion(output, y_batch)
        grad_f = torch.autograd.grad(loss, inputs=params, create_graph=True)

        grad_times_vec = 0.0
        for g, v in zip(grad_f, vec):
            grad_times_vec += (g * v).sum()

        grad_times_vec.backward()
