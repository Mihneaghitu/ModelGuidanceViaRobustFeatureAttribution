"""
This file implements the robust regularisation term from the paper 'Robust Explanation Constraints for Neural Networks'.
Nothing about this is 'certified', but we can use it for robust (but un-certified) pre-training.
"""

import torch
from abstract_gradient_training import interval_arithmetic


def gradient_interval_regularizer(
    model: torch.nn.Sequential,
    batch: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: str,
    epsilon: float,
    model_epsilon: float,
    return_grads: bool = False,
) -> float:
    """
    Compute an interval over the gradients of the loss with respect to the inputs to the network. Then compute the norm
    of this input gradient interval and return it to be used as a regularization term. This can be used for robust
    (but un-certified) pre-training.

    Args:
        model (torch.nn.Sequential): The neural network model.
        batch (torch.Tensor): The input data batch, should have shape [batchsize x ... ].
        labels (torch.Tensor): The labels for the input data batch, should have shape [batchsize].
        loss_fn (str): Only "binary_cross_entropy" supported for now.
        epsilon (float): The training time input perturbation budget.
        model_epsilon (float): The training time model perturbation budget.
        return_grads (bool): Whether to return the gradients directly, instead of the regularization term.

    Returns:
        float: The regularization term.
    """
    # we only support binary cross entropy loss for now
    if loss_fn != "binary_cross_entropy":
        raise ValueError(f"Unsupported loss function: {loss_fn}")
    elif not isinstance(model[-1], torch.nn.Sigmoid):
        raise ValueError(f"Expected last layer to be sigmoid for binary cross entropy loss, got {model[-1]}")
    else:
        modules = list(model.modules())[1:-1]

    # propagate through the forward pass
    intermediate = [(batch - epsilon, batch + epsilon)]
    for module in modules:
        intermediate.append(propagate_module_forward(module, *intermediate[-1], model_epsilon))
        interval_arithmetic.validate_interval(*intermediate[-1], msg=f"forward pass {module}")

    # propagate through the loss function
    logits_l, logits_u = intermediate[-1]
    logits_l, logits_u = logits_l.squeeze(1), logits_u.squeeze(1)
    dl_l, dl_u = torch.sigmoid(logits_l) - labels, torch.sigmoid(logits_u) - labels
    interval_arithmetic.validate_interval(dl_l, dl_u, msg="loss gradient")

    # propagate through the backward pass
    grads = [dl_l]
    for i in range(len(modules) - 1, -1, -1):
        dl_l, dl_u = propagate_module_backward(modules[i], dl_l, dl_u, *intermediate[i], model_epsilon)
        interval_arithmetic.validate_interval(dl_l, dl_u, msg=f"backward pass {modules[i]}")
        grads.append(dl_l)

    if return_grads:
        grads.reverse()
        return grads
    return torch.norm(dl_u - dl_l, p=2)


def propagate_module_forward(
    module: torch.nn.Module, x_l: torch.Tensor, x_u: torch.Tensor, model_epsilon: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Propagate the input interval through the given nn module.

    Args:
        module (torch.nn.Module): The module to propagate through.
        x_l (torch.Tensor): Lower bound on the input.
        x_u (torch.Tensor): Upper bound on the input.
        model_epsilon (float): The model perturbation budget.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Lower and upper bounds on the output of the module.
    """
    if isinstance(module, torch.nn.Linear):
        W, b = module.weight, module.bias
        b = torch.zeros(W.size(0), 1, device=x_l.device, dtype=x_l.dtype) if b is None else b.unsqueeze(-1)
        W_l, W_u = W - model_epsilon, W + model_epsilon
        b_l, b_u = b - model_epsilon, b + model_epsilon
        if x_l.dim() == 2:
            x_l, x_u = x_l.unsqueeze(-1), x_u.unsqueeze(-1)
        x_l, x_u = interval_arithmetic.propagate_affine(x_l, x_u, W_l, W_u, b_l, b_u)
    elif isinstance(module, torch.nn.ReLU):
        x_l, x_u = torch.relu(x_l), torch.relu(x_u)
    elif isinstance(module, torch.nn.Conv2d):
        W, b = module.weight, module.bias
        W_l, W_u = W - model_epsilon, W + model_epsilon
        b_l, b_u = b - model_epsilon, b + model_epsilon
        dilation, stride, padding = module.dilation, module.stride, module.padding
        x_l, x_u = interval_arithmetic.propagate_conv2d(
            x_l, x_u, W_l, W_u, b_l, b_u, stride=stride, padding=padding, dilation=dilation
        )
    elif isinstance(module, torch.nn.Flatten):
        x_l, x_u = module(x_l), module(x_u)
    else:
        raise ValueError(f"Unsupported module type: {type(module)}")
    interval_arithmetic.validate_interval(x_l, x_u, msg=f"module {module}")
    return x_l, x_u


def propagate_module_backward(
    module: torch.nn.Module,
    dl_l: torch.Tensor,
    dl_u: torch.Tensor,
    x_l: torch.Tensor,
    x_u: torch.Tensor,
    model_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Back-propagate the gradient interval through the given nn module.

    Args:
        module (torch.nn.Module): The module to propagate through.
        dl_l (torch.Tensor): Lower bound on the gradient leading to this module.
        dl_u (torch.Tensor): Upper bound on the gradient leading to this module.
        x_l (torch.Tensor): Lower bound on the input to the module.
        x_u (torch.Tensor): Upper bound on the input to the module.
        model_epsilon (float): The model perturbation budget.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Lower and upper bounds on the gradient leading to the input of the module.
    """
    interval_arithmetic.validate_interval(dl_l, dl_u, msg="dl input")
    if isinstance(module, torch.nn.Linear):
        W = module.weight
        W_l, W_u = W - model_epsilon, W + model_epsilon
        dl_l, dl_u = interval_arithmetic.propagate_matmul(dl_l, dl_u, W_l, W_u)
    elif isinstance(module, torch.nn.ReLU):
        x_l, x_u = x_l.reshape(dl_l.size()), x_u.reshape(dl_u.size())
        dl_l, dl_u = interval_arithmetic.propagate_elementwise(dl_l, dl_u, (x_l > 0).float(), (x_u > 0).float())
    elif isinstance(module, torch.nn.Conv2d):
        W, dilation, stride, padding = module.weight, module.dilation, module.stride, module.padding
        W_l, W_u = W - model_epsilon, W + model_epsilon
        dl_l, dl_u = interval_arithmetic.propagate_conv2d(
            dl_l, dl_u, W_l, W_u, stride=stride, padding=padding, dilation=dilation, transpose=True
        )
    elif isinstance(module, torch.nn.Flatten):
        dl_l, dl_u = torch.reshape(dl_l, x_l.size()), torch.reshape(dl_u, x_u.size())
    else:
        raise ValueError(f"Unsupported module type: {type(module)}")
    interval_arithmetic.validate_interval(dl_l, dl_u, msg=f"module {module}")
    return dl_l, dl_u


if __name__ == "__main__":
    # test the gradient calculations vs torch autograd
    import sys

    sys.path.append("..")
    from datasets.oct_mnist import get_dataloaders  # pylint: disable=import-error
    from deepmind import DeepMindSmall

    # define model, dataset and optimizer
    device = "cuda:0"
    torch.manual_seed(0)
    test_model = DeepMindSmall(1, 1)
    dl, _ = get_dataloaders(64)
    criterion = torch.nn.BCELoss(reduction="sum")
    test_model = test_model.to(device)

    # get the input data and pass through the model and loss
    test_batch, test_labels = next(iter(dl))
    test_batch, test_labels = test_batch.to(device), test_labels.to(device)
    test_batch.requires_grad = True
    intermediates = [test_batch]
    for layer in list(test_model.modules())[1:]:
        intermediates.append(layer(intermediates[-1]))
        intermediates[-1].retain_grad()
    loss = criterion(intermediates[-1].squeeze().float(), test_labels.squeeze().float())
    loss.backward()
    autograds = [inter.grad for inter in intermediates]

    # pass the data through our gradient interval regularizer with zero epsilon, which should match the exact grads
    custom_grads = gradient_interval_regularizer(
        test_model, test_batch, test_labels, "binary_cross_entropy", 0.0, 0.0, return_grads=True
    )

    for j in range(len(custom_grads)):
        print(f"Layer {j}: matching gradients = {torch.allclose(custom_grads[j], autograds[j])}")

    # gradient regularization term:
    reg = gradient_interval_regularizer(test_model, test_batch, test_labels, "binary_cross_entropy", 0.1, 0.0)
    print("Regularization term (input perturbation):", reg)
    reg = gradient_interval_regularizer(test_model, test_batch, test_labels, "binary_cross_entropy", 0.0, 0.001)
    print("Regularization term (model perturbation):", reg)
