"""Functions that perform the nominal forward and backward passes through the network."""

import torch
import torch.nn.functional as F


def nominal_forward_pass(x0: torch.Tensor, params: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    Perform the forward pass through the network with the given nominal parameters.

    Args:
        x0 (torch.Tensor): [batchsize x input_dim x 1] tensor of inputs to the network
        params (list[torch.Tensor]): List of the nominal parameters of the network [W1, b1, ..., Wn, bn].

    Returns:
        activations (list[torch.Tensor]): List of tensors [x0, x1, ..., xn] where x0 is the input, xn is the logit and
            x1, ..., xn-1 are the (pre-relu) intermediate activations. Each tensor is of shape [batchsize x dim x 1].
    """

    assert len(x0.shape) == 3  # this function expects a batched input
    # we want to be able to accept biases as either 1 or 2 dimensional tensors
    params = [p if len(p.shape) == 2 else p.unsqueeze(-1) for p in params]
    activations = [x0]
    for i, (wk, bk) in enumerate(zip(params[::2], params[1::2])):
        if i == 0:
            activations.append(wk @ x0 + bk)  # input doesn't go through relu
        else:
            activations.append(wk @ F.relu(activations[-1]) + bk)
    return activations


def nominal_backward_pass(
    dL: torch.Tensor, params: list[torch.Tensor], activations: list[torch.Tensor], with_input_grad: bool = False
) -> list[torch.Tensor]:
    """
    Perform the backward pass through the network with nominal parameters given dL is the first partial derivative of
    the loss with respect to the logits.

    Args:
        dL (torch.Tensor): Tensor of shape [batchsize x output_dim x 1] representing the gradient of the loss with
            respect to the logits of the network.
        params (list[torch.Tensor]): List of the nominal parameters of the network [W1, b1, ..., Wn, bn].
        activations (list[torch.Tensor]): List of tensors [x0, x1, ..., xn] where x0 is the input, xn is the logit and
            x1, ..., xn-1 are the (pre-relu) intermediate activations. Each tensor is of shape [batchsize x dim x 1].
    Returns:
        list[torch.Tensor]: List of gradients of the network [dW1, db1, ..., dWm, dbm]
    """
    # convert the pre-relu activations to post-relu activations
    activations = [activations[0]] + [F.relu(x) for x in activations[1:-1]] + [activations[-1]]  # x0, x1, ..., xL-1, xL

    # we want to be able to accept biases as either 1 or 2 dimensional tensors
    params = [p if len(p.shape) == 2 else p.unsqueeze(-1) for p in params]
    dL = dL if len(dL.shape) == 3 else dL.unsqueeze(-1)
    W = params[::2]
    dW = dL @ activations[-2].transpose(-2, -1)
    grads = [dL, dW]

    # compute gradients for each layer
    lim = -1 if with_input_grad else 0
    for i in range(len(W) - 1, lim, -1):
        dL = W[i].T @ dL
        if i == 0:
            grads.append(dL)
            break
        dL = dL * torch.sign(activations[i])
        dW = dL @ activations[i - 1].transpose(-2, -1)
        grads.append(dL)
        grads.append(dW)

    grads.reverse()
    return grads

def nominal_input_gradient(
    params: list[torch.Tensor], activations: list[torch.Tensor], d_l: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the gradient of the loss with respect to the input.

    Args:
        params (list[torch.Tensor]): List of the nominal parameters of the network [W1, b1, ..., Wn, bn].
        activations (list[torch.Tensor]): List of tensors [x0, x1, ..., xn] where x0 is the input, xn is the logit and
                                          x1, ..., xn-1 are the (pre-relu) intermediate activations. Each tensor is of
                                          shape [batchsize x dim x 1].
        d_l (torch.Tensor): Tensor of shape [batchsize x output_dim x 1] representing the gradient of the output of the network with
                           respect to the logits of the network.

    Returns:
        torch.Tensor: Tensor of shape [batchsize x input_dim x 1] representing the gradient of the network's output with respect to
                     the input.
    """
    # convert the pre-relu activations to post-relu activations
    activations = [activations[0]] + [F.relu(x) for x in activations[1:-1]] + [activations[-1]]  # x0, x1, ..., xL-1, xL

    # we want to be able to accept biases as either 1 or 2 dimensional tensors
    params = [p if len(p.shape) == 2 else p.unsqueeze(-1) for p in params]
    d_l = d_l if len(d_l.shape) == 3 else d_l.unsqueeze(-1)
    w = params[::2]

    # compute gradients for each layer
    for i in range(len(w) - 1, 0, -1):
        d_l = d_l @ w[i] # now we have the derivative with respect to the pre-relu activations
        # backprop through relu
        d_l = d_l * torch.sign(activations[i].transpose(-2, -1))

    d_x0 = d_l @ w[0]

    return d_x0


def nominal_forward_pass_with_conv(x0: torch.Tensor, params: list[torch.Tensor]) -> list[torch.Tensor]:
    assert len(x0.shape) == 4  # this function expects a batched input, with channels and dims: B x C x H x W
    # we want to be able to accept biases as either 1 or 2 dimensional tensors
    params = [p if len(p.shape) == 3 else p.unsqueeze(-1) for p in params]
    activations = [x0]
    for i, (wk, bk) in enumerate(zip(params[::2], params[1::2])):
        print(f"conv weights shape: {wk.shape}")
        print(f"conv bias shape: {bk.shape}")
        if i == 0:
            activations.append(wk @ x0 + bk)  # input doesn't go through relu
        else:
            activations.append(wk @ F.relu(activations[-1]) + bk)
    return activations


def nominal_backward_pass_with_conv(
    dL: torch.Tensor, params: list[torch.Tensor], activations: list[torch.Tensor], with_input_grad: bool = False
) -> list[torch.Tensor]:
    # convert the pre-relu activations to post-relu activations
    activations = [activations[0]] + [F.relu(x) for x in activations[1:-1]] + [activations[-1]]  # x0, x1, ..., xL-1, xL

    # we want to be able to accept biases as either 1 or 2 dimensional tensors
    params = [p if len(p.shape) == 3 else p.unsqueeze(-1) for p in params]
    dL = dL if len(dL.shape) == 4 else dL.unsqueeze(-1)
    W = params[::2]
    dW = dL @ activations[-2].transpose(-2, -1)
    grads = [dL, dW]

    # compute gradients for each layer
    lim = -1 if with_input_grad else 0
    for i in range(len(W) - 1, lim, -1):
        dL = W[i].T @ dL
        if i == 0:
            grads.append(dL)
            break
        dL = dL * torch.sign(activations[i])
        dW = dL @ activations[i - 1].transpose(-2, -1)
        grads.append(dL)
        grads.append(dW)

    grads.reverse()
    return grads