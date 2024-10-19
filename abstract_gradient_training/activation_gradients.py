
"""
Functions to compute the partial derivative of the network output function with respect to its logits.
"""
import torch
import torch.nn.functional as F


def bound_logits_derivative(logits: torch.Tensor, loss_fn: str) -> torch.Tensor:
    match loss_fn:
        case 'cross_entropy':
            return softmax_gradient(logits)
        case 'binary_cross_entropy':
            return sigmoid_gradient(logits)
        case _:
            raise ValueError(f"Unsupported loss function: {loss_fn}")


# TODO: check dims of every other function apart from softmax
def softmax_gradient(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute the gradient of the softmax function with respect to the logits.
    Args:
        logits (torch.Tensor): Tensor of shape [batchsize x output_dim x 1] representing the logits of the network.
    Returns:
        torch.Tensor: Tensor of shape [batchsize x output_dim x output_dim] representing the gradient of the softmax
                      function with respect to the logits.
    """
    # compute the softmax function
    # logits shape is [batchsize x output_dim x 1]
    logits = logits.squeeze(-1)
    y_hat = F.softmax(logits, dim=1) # [500, 10, 10]

    # compute the gradient of the softmax function given that y_hat has dimension batchsize x output_dim
    return torch.diag_embed(y_hat) - y_hat.unsqueeze(-1) @ y_hat.unsqueeze(1)

def sigmoid_gradient(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute the gradient of the sigmoid function with respect to the logits.
    Args:
        logits (torch.Tensor): Tensor of shape [batchsize x output_dim x 1] representing the logits of the network.
    Returns:
        torch.Tensor: Tensor of shape [batchsize x output_dim x output_dim] representing the gradient of the sigmoid
                      function with respect to the logits.
    """
    # compute the sigmoid function
    # logits shape is [batchsize x output_dim x 1]
    y_hat = torch.sigmoid(logits)
    # compute the gradient of the sigmoid function
    return torch.diag_embed(y_hat) - y_hat.unsqueeze(-1) @ y_hat.unsqueeze(1)

def relu_gradient(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute the gradient of the ReLU function with respect to the logits.
    Args:
        logits (torch.Tensor): Tensor of shape [batchsize x output_dim x 1] representing the logits of the network.
    Returns:
        torch.Tensor: Tensor of shape [batchsize x output_dim x output_dim] representing the gradient of the ReLU
                      function with respect to the logits.
    """
    # compute the gradient of the ReLU function
    return torch.diag_embed((logits > 0).float())

def identity_gradient(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute the gradient of the identity function with respect to the logits.
    Args:
        logits (torch.Tensor): Tensor of shape [batchsize x output_dim x 1] representing the logits of the network.
    Returns:
        torch.Tensor: Tensor of shape [batchsize x output_dim x output_dim] representing the gradient of the identity
                      function with respect to the logits.
    """
    return torch.eye(logits.size(1)).unsqueeze(0).expand(logits.size(0), -1, -1)
