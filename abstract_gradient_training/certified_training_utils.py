"""Helper functions for certified training."""

from __future__ import annotations
from collections.abc import Iterator

import logging
import torch
from torch.utils.data import DataLoader

from abstract_gradient_training import interval_arithmetic
from abstract_gradient_training.configuration import AGTConfig


LOGGER = logging.getLogger(__name__)


def grads_helper(
    batch_l: torch.Tensor,
    batch_u: torch.Tensor,
    labels: torch.Tensor,
    param_l: list[torch.Tensor],
    param_u: list[torch.Tensor],
    config: AGTConfig,
    label_poison: bool = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Helper function to calculate bounds on the gradient of the loss function with respect to all parameters given the
    input and parameter bounds.

    Args:
        batch_l (torch.Tensor): [fragsize x input_dim x 1] tensor of inputs to the network.
        batch_u (torch.Tensor): [fragsize x input_dim x 1] tensor of inputs to the network.
        labels (torch.Tensor): [fragsize, ] tensor of labels for the inputs.
        param_l (list[torch.Tensor]): List of the lower bound parameters of the network [W1, b1, ..., Wn, bn].
        param_u (list[torch.Tensor]): List of the upper bound parameters of the network [W1, b1, ..., Wn, bn].
        config (AGTConfig): Configuration object for the abstract gradient training module.
        label_poison (bool, optional): Boolean flag to indicate if the labels are being poisoned. Defaults to False.

    Returns:
        tuple[list[torch.Tensor], list[torch.Tensor]]: List of lower and upper bounds on the gradients.
    """
    labels = labels.squeeze()
    assert labels.dim() == 1, "Labels must be of shape (batchsize, )"
    # get config parameters
    bound_kwargs = config.bound_kwargs
    loss_bound_fn = config.loss_bound_fn
    forward_bound_fn = config.forward_bound_fn
    backward_bound_fn = config.backward_bound_fn
    label_epsilon = config.label_epsilon if label_poison else 0.0
    k_label_poison = config.label_k_poison if label_poison else 0
    poison_target = config.poison_target if label_poison else -1
    # forward pass through the network with bounds
    activations_l, activations_u = forward_bound_fn(param_l, param_u, batch_l, batch_u, **bound_kwargs)
    # calculate the first partial derivative of the loss function
    # (pass logit_u in as a dummy for logit_n and ignore dL_n)
    dL_l, dL_u, _ = loss_bound_fn(
        activations_l[-1],  # logit_l
        activations_u[-1],
        activations_u[-1],  # logit_u
        labels,
        k_label_poison=k_label_poison,
        label_epsilon=label_epsilon,
        poison_target=poison_target,
    )
    # compute backwards pass through the network with bounds
    grad_min, grad_max = backward_bound_fn(dL_l, dL_u, param_l, param_u, activations_l, activations_u, **bound_kwargs)

    return grad_min, grad_max


def break_condition(evaluation: tuple[float, float, float]) -> bool:
    """
    Check whether to terminate the certified training loop based on the bounds on the test metric (MSE or Accuracy).

    Args:
        evaluation: tuple of the (worst case, nominal case, best case) evaluation of the test metric.

    Returns:
        bool: True if the training should stop, False otherwise.
    """
    if evaluation[0] <= 0.03 and evaluation[2] >= 0.97:  # worst case accuracy bounds too loose
        LOGGER.warning("Early stopping due to loose bounds")
        return True
    if evaluation[0] >= 1e2:  # worst case MSE too large
        LOGGER.warning("Early stopping due to loose bounds")
        return True
    return False


def get_progress_message(
    network_eval: tuple[float, float, float], param_l: list[torch.Tensor], param_u: list[torch.Tensor]
) -> str:
    """
    Generate a progress message for the certified training loop.

    Args:
        network_eval (tuple[float, float, float]): (worst case, nominal case, best case) evaluation of the test metric.
        param_l (list[torch.Tensor]): List of the lower bound parameters of the network [W1, b1, ..., Wn, bn].
        param_u (list[torch.Tensor]): List of the upper bound parameters of the network [W1, b1, ..., Wn, bn].

    Returns:
        str: Progress message for the certified training loop.
    """
    msg = (
        f"Network eval bounds=({network_eval[0]:<4.2g}, {network_eval[1]:<4.2g}, {network_eval[2]:<4.2g}), "
        f"W0 Bound={(param_l[0] - param_u[0]).norm():.3} "
    )

    return msg


def get_parameters(model: torch.nn.Sequential) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    Get the parameters of only the dense layers of the pytorch model. This function assumes that all dense layers
    are at the end of the model separated by activation functions only.

    Args:
        model (torch.nn.Sequential): Pytorch model to extract the parameters from.

    Returns:
        tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]: Tuple of lists of the network parameters
            [W1, b1, ..., Wn, bn] for the network, the lower bounds of the parameters, and the upper bounds of the
            parameters.
    """
    param_n = [(l.weight, l.bias) for l in model.modules() if isinstance(l, torch.nn.Linear)]  # get linear params
    param_n = [item for sublist in param_n for item in sublist]  # flatten the list
    param_n = [t if len(t.shape) == 2 else t.unsqueeze(-1) for t in param_n]  # reshape bias to [n x 1] instead of [n]
    param_n = [t.detach().clone() for t in param_n]
    param_l = [p.clone() for p in param_n]
    param_u = [p.clone() for p in param_n]
    return param_n, param_l, param_u

def get_nominal_parameters(model: torch.nn.Sequential) -> list[torch.Tensor]:
    param_n = [(l.weight, l.bias) for l in model.modules() if isinstance(l, torch.nn.Linear) or isinstance(l, torch.nn.Conv2d)]  # get linear and conv params
    param_n = [item for sublist in param_n for item in sublist]  # flatten the list
    param_n = [t if len(t.shape) == 2 else t.unsqueeze(-1) for t in param_n]  # reshape bias to [n x 1] instead of [n]

    return param_n

def propagate_clipping(
    x_l: torch.Tensor, x: torch.Tensor, x_u: torch.Tensor, gamma: float, method: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Propagate the input through a clipping operation. This function is used to clip the gradients in the
    DP-SGD algorithm.

    Args:
        x_l (torch.Tensor): Lower bound of the input tensor.
        x_u (torch.Tensor): Upper bound of the input tensor.
        gamma (float): Clipping parameter.
        method (str): Clipping method, one of ["clamp", "norm"].

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of the lower, nominal and upper bounds of the clipped
            input tensor.
    """
    if method == "clamp":
        x_l = torch.clamp(x_l, -gamma, gamma)
        x = torch.clamp(x, -gamma, gamma)
        x_u = torch.clamp(x_u, -gamma, gamma)
    elif method == "norm":
        interval_arithmetic.validate_interval(x_l, x_u, msg="input")
        # compute interval over the norm of the input interval
        norms = x.flatten(1).norm(2, dim=1)
        norms_l, norms_u = interval_arithmetic.propagate_norm(x_l, x_u, p=2)
        interval_arithmetic.validate_interval(norms_l, norms_u, msg="norm")
        # compute an interval over the clipping factor
        clip_factor = (gamma / (norms + 1e-6)).clamp(max=1.0)
        clip_factor_l = (gamma / (norms_u + 1e-6)).clamp(max=1.0)
        clip_factor_u = (gamma / (norms_l + 1e-6)).clamp(max=1.0)
        interval_arithmetic.validate_interval(clip_factor_l, clip_factor_u, msg="clip factor")
        # compute an interval over the clipped input
        x_l, x_u = interval_arithmetic.propagate_elementwise(
            x_l, x_u, clip_factor_l.view(-1, 1, 1), clip_factor_u.view(-1, 1, 1)
        )
        x = x * clip_factor.view(-1, 1, 1)
        interval_arithmetic.validate_interval(x_l, x_u, msg="clipped input")
    else:
        raise ValueError(f"Clipping method {method} not recognised.")
    return x_l, x, x_u


def propagate_conv_layers(
    x: torch.Tensor, model: torch.nn.Sequential, epsilon: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Propagate an input batch through the convolutional layers of a model. Here we assume that the conv layers are all
    at the start of the network with ReLU activations after each one.

    Args:
        x (torch.Tensor): [batchsize x input_dim x 1] tensor of inputs to the network.
        model (torch.nn.Sequential): Pytorch model to extract the parameters from.
        epsilon (float): Epsilon value for the interval propagation.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tuple of the lower bounds and upper bounds of the output of the
                                           convolutional layers of the network.
    """
    # get the parameters of the conv layers
    conv_layers = [l for l in model.modules() if isinstance(l, torch.nn.Conv2d)]
    conv_parameters = []
    for l in conv_layers:
        bias = l.bias.detach() if l.bias is not None else l.bias
        conv_parameters.append((l.weight.detach(), bias, l.stride, l.padding, l.dilation))
    # propagate the input through the conv layers
    x_l, x_u = x - epsilon, x + epsilon
    for W, b, stride, padding, dilation in conv_parameters:
        x_l, x_u = interval_arithmetic.propagate_conv2d(
            x_l, x_u, W, W, b, b, stride=stride, padding=padding, dilation=dilation
        )
        x_l, x_u = torch.nn.functional.relu(x_l), torch.nn.functional.relu(x_u)
    x_l = x_l.flatten(start_dim=1)
    x_u = x_u.flatten(start_dim=1)
    return x_l.unsqueeze(-1), x_u.unsqueeze(-1)


def dataloader_wrapper(dl_train: DataLoader, n_epochs: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """
    Return a new generator that iterates over the training dataloader for a fixed number of epochs.
    This includes a check to ensure that each batch is full and ignore any incomplete batches.
    Note that we assume the first batch is full.
    """
    assert len(dl_train) > 0, "Dataloader is empty!"
    if len(dl_train) == 1:
        LOGGER.warning("Dataloader has only one batch, effective batchsize may be smaller than expected.")
    # batchsize variable will be initialised in the first iteration
    full_batchsize = None
    # loop over epoch
    for n in range(n_epochs):
        LOGGER.info("Starting epoch %s", n + 1)
        for t, (batch, labels) in enumerate(dl_train):
            # initialise the batchsize variable if this is the first iteration
            if full_batchsize is None:
                full_batchsize = batch.size(0)
                LOGGER.debug("Initialising dataloader batchsize to %s", full_batchsize)
            # check the batch is the correct size, otherwise skip it
            if batch.size(0) != full_batchsize:
                LOGGER.debug(
                    "Skipping batch %s in epoch %s (expected batchsize %s, got %s)",
                    t + 1,
                    n + 1,
                    full_batchsize,
                    batch.size(0),
                )
                continue
            # return the batches for this iteration
            yield batch, labels


def dataloader_pair_wrapper(
    dl_train: DataLoader, dl_clean: DataLoader | None, n_epochs: int, dtype: torch.dtype
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]]:
    """
    Return a new generator that iterates over the training dataloaders for a fixed number of epochs.
    For each combined batch, we return one batch from the clean dataloader and one batch from the poisoned dataloader.
    This includes a check to ensure that each batch is full and ignore any incomplete batches.
    Note that we assume the first batch is full.
    """
    assert len(dl_train) > 0, "Dataloader is empty!"
    # this is the max number of batches, if none are skipped due to being incomplete
    max_batches_per_epoch = len(dl_train) if dl_clean is None else min(len(dl_train), len(dl_clean))
    if max_batches_per_epoch == 1:
        LOGGER.warning("Dataloader has only one batch, effective batchsize may be smaller than expected.")
    # batchsize variable will be initialised in the first iteration
    full_batchsize = None
    # loop over epochs
    for n in range(n_epochs):
        LOGGER.info("Starting epoch %s", n + 1)
        # handle the case where there is no clean dataloader by returning dummy values
        if dl_clean is None:
            data_iterator = (((b, l), (None, None)) for b, l in dl_train)
        else:
            # note that zip will stop at the shortest iterator
            data_iterator = zip(dl_train, dl_clean)
        for t, ((batch, labels), (batch_clean, labels_clean)) in enumerate(data_iterator):
            # check the length of this batch
            batch_len = batch.size(0)
            if batch_clean is not None:
                batch_len += batch_clean.size(0)
            # convert to expected dtype
            batch = batch.to(dtype)
            batch_clean = batch_clean.to(dtype) if batch_clean is not None else None
            batchsize = batch.size(0) if batch_clean is None else batch.size(0) + batch_clean.size(0)
            # initialise the batchsize variable if this is the first iteration
            if full_batchsize is None:
                full_batchsize = batchsize
                LOGGER.debug("Initialising dataloader batchsize to %s", full_batchsize)
            # check the batch is the correct size, otherwise skip it
            if batchsize != full_batchsize:
                LOGGER.debug(
                    "Skipping batch %s in epoch %s (expected batchsize %s, got %s)",
                    t + 1,
                    n + 1,
                    full_batchsize,
                    batch_len,
                )
                continue
            # return the batches for this iteration
            yield batch, labels, batch_clean, labels_clean
