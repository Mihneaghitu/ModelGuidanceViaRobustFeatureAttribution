"""
This file implements the robust regularisation term from the paper 'Robust Explanation Constraints for Neural Networks'.
Nothing about this is 'certified', but we can use it for robust (but un-certified) pre-training.
"""

import torch

import abstract_gradient_training.certified_training_utils as ct_utils
from abstract_gradient_training import interval_arithmetic
from abstract_gradient_training.activation_gradients import bound_logits_derivative
from abstract_gradient_training.bounds.loss_gradients import \
    bound_loss_function_derivative
from abstract_gradient_training.nominal_pass import (nominal_backward_pass,
                                                     nominal_forward_pass)
import copy

#TODO it seems that for r3, ibp_ex and pgd_ex the smoothing does all the job? Is there something wrong in my code?
def input_gradient_interval_regularizer(
    model: torch.nn.Sequential,
    batch: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: str,
    epsilon: float,
    model_epsilon: float,
    return_grads: bool = False,
    regularizer_type: str = "std",
    batch_masks: torch.Tensor = None,
    has_conv: bool = False,
    device: str = "cuda:0",
    weight_reg_coeff: float = 0.0,
) -> torch.Tensor | list[tuple[torch.Tensor]]:
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
        regularizer_type (str): The type of regularizer to use. Can be "grad_cert", "r4", "r3", "std" or "ibp_ex".
        batch_masks (torch.Tensor): The salient image masks for the input data batch, should match the shape of the input.
        has_conv (bool): Whether the model has convolutional layers.
        device (str): The device to use for computation.

    Returns:
        float | list[torch.Tensor]: The regularization term or the gradient lower bounds.
    """
    # we only support binary cross entropy loss for now
    assert loss_fn in ["binary_cross_entropy", "cross_entropy"], f"Unsupported loss function: {loss_fn}"
    assert regularizer_type in ["grad_cert", "r4", "r3", "std", "ibp_ex", "ibp_ex+r3"], f"Unsupported regularizer type: {regularizer_type}"
    assert batch_masks is not None
    #! =========================== Checkpoint to make training faster ===========================
    if regularizer_type == "std":
        return 0
    #! ==================================== End of Checkpoint ===================================
    modules = list(model.modules())
    assert isinstance(modules[-1], torch.nn.Sigmoid) or isinstance(modules[-1], torch.nn.Softmax)
    last_layer_act_func = modules[-1]
    criterion = torch.nn.BCELoss() if loss_fn == "binary_cross_entropy" else torch.nn.CrossEntropyLoss()
    # remove the first module (copy of model) and the last module (sigmoid or softmax)
    # if the first module is a DataParallel, remove the first two instead
    modules = modules[2:-1] if isinstance(modules[0], torch.nn.DataParallel) else modules[1:-1]
    params_n_dense = ct_utils.get_parameters(model)[0]
    # do a forward pass without gradients to be able to call the loss gradient bounding functions in the agt module
    with torch.no_grad():
        logits_n = batch
        for module in modules:
            logits_n = module(logits_n)
        logits_n = logits_n.unsqueeze(-1)

    # ================================= BOUNDS COMPUTATION =================================
    intermediate = None
    if regularizer_type in ["r4", "ibp_ex", "ibp_ex+r3"]:
        intermediate = [(batch - epsilon * batch_masks, batch + epsilon * batch_masks)]
    else:
        intermediate = [(batch - epsilon, batch + epsilon)]
    # This is needed so that we can use the same uniform interface of the bounds to get the input gradient
    intermediate_nominal = [(batch, batch)]
    # propagate through the forward pass
    for module in modules:
        intermediate.append(propagate_module_forward(module, *intermediate[-1], model_epsilon))
        intermediate_nominal.append(propagate_module_forward(module, *intermediate_nominal[-1], 0))
        interval_arithmetic.validate_interval(*intermediate[-1], msg=f"forward pass {module}")

    logits_l, logits_u = intermediate[-1]
    #! =========================== Checkpoint to make training faster ===========================
    weight_sum = torch.tensor(0).to(device, dtype=torch.float32).requires_grad_()
    for module in modules:
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            weight_sum = weight_sum + torch.sum(module.weight ** 2)
    l2_reg = weight_reg_coeff * weight_sum
    match regularizer_type:
        case "ibp_ex":
            y_bar = None
            if loss_fn == "binary_cross_entropy":
                y_bar = last_layer_act_func(logits_l)
            else: # cross entropy
                labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=modules[-1].out_features)
                # squeeze because logits are [batchsize x output_dim x 1]
                y_bar = last_layer_act_func(logits_l).squeeze() * labels_one_hot + last_layer_act_func(logits_u).squeeze() * (1 - labels_one_hot)
            return l2_reg + criterion(y_bar.squeeze(), labels)
    #! ==================================== End of Checkpoint ===================================

    # propagate through the loss function
    dl_l, dl_u, dl_n = bound_loss_function_derivative(loss_fn, logits_l, logits_u, logits_n, labels)
    dl_l, dl_u = dl_l.squeeze(-1), dl_u.squeeze(-1)
    interval_arithmetic.validate_interval(dl_l, dl_u, msg="loss gradient")

    # propagate through the backward pass
    grads = [(dl_l, dl_u)]
    for i in range(len(modules) - 1, -1, -1):
        dl_l, dl_u = propagate_module_backward(modules[i], dl_l, dl_u, *intermediate[i], model_epsilon)
        interval_arithmetic.validate_interval(dl_l, dl_u, msg=f"backward pass {modules[i]}")
        # This will happen when we have a Flatten() module at the beginning of the model
        if i == 0 and dl_l.shape != batch_masks.shape:
            dl_l, dl_u = dl_l.reshape(batch_masks.shape), dl_u.reshape(batch_masks.shape)
        grads.append((dl_l, dl_u))


    # ================================= INPUT GRAD COMPUTATION =================================
    # Need to unsqueeze the batch dimension if not conv to satisfy the quirks of this function (experimented on isic and decoy_mnist)
    input_grad = None
    if has_conv:
        if loss_fn == "cross_entropy":
            dl_n = dl_n.squeeze(-1)
        for i in range(len(modules) - 1, -1, -1):
            dl_n, dl_n_1 = propagate_module_backward(modules[i], dl_n, dl_n, *intermediate_nominal[i], 0) # 0 model epsilon
            interval_arithmetic.validate_interval(dl_l, dl_u, msg=f"backward pass {modules[i]}")
            assert torch.allclose(dl_n, dl_n_1)
        input_grad = dl_n
    else:
        activations_n = nominal_forward_pass(batch.flatten(start_dim=1).unsqueeze(-1), params_n_dense)
        input_grad = nominal_backward_pass(dl_n, params_n_dense, activations_n, with_input_grad=True)[0]
    # This will happen when we have a Flatten() module at the beginning of the model
    if input_grad.shape != batch_masks.shape:
        input_grad = input_grad.reshape(batch_masks.shape)
    grads.append((input_grad, None))

    if return_grads:
        grads.reverse()
        return grads

    # pre-compute the regularization term for code clarity
    match regularizer_type:
        case "grad_cert":
            # Loss for the interval regularizer
            return torch.norm(dl_u - dl_l, p=2) / dl_l.nelement()
        case "r4":
            # Loss for R4
            return torch.sum(torch.abs(torch.mul(dl_l, batch_masks)) +
            torch.abs(torch.mul(dl_u, batch_masks))) / dl_l.nelement()
        case "r3":
            # return the gradient of the loss w.r.t. the input squared and summed
            # input_grad_logits should now be of shape [batchsize x input_dim]
            reg_term = torch.tensor(0).to(device, dtype=torch.float32).requires_grad_()
            # make sure the input_grad is of the same shape as the input
            reg_term = reg_term + torch.sum((input_grad.squeeze() * batch_masks) ** 2) / input_grad.nelement()
            return reg_term + l2_reg
        case "ibp_ex+r3":
            # ibp_ex term
            y_bar = None
            if loss_fn == "binary_cross_entropy":
                y_bar = last_layer_act_func(logits_l)
            else: # cross entropy
                labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=modules[-1].out_features)
                # squeeze because logits are [batchsize x output_dim x 1]
                y_bar = last_layer_act_func(logits_l).squeeze() * labels_one_hot + last_layer_act_func(logits_u).squeeze() * (1 - labels_one_hot)
            # r3 term
            reg_term = torch.tensor(0).to(device, dtype=torch.float32).requires_grad_()
            reg_term = reg_term + torch.sum((input_grad.squeeze() * batch_masks) ** 2) / input_grad.nelement()
            # combine
            return l2_reg + criterion(y_bar.squeeze(), labels) + reg_term

def smooth_gradient_regularizer(
    batch: torch.Tensor,
    labels: torch.Tensor,
    model: torch.nn.Module,
    batch_masks: torch.Tensor,
    criterion: torch.nn.Module,
    epsilon: float,
    regularizer_type: str = "smooth_r3",
    device: str = "cuda:0",
    weight_reg_coeff: float = 0.0
) -> torch.Tensor:
    assert regularizer_type in ["smooth_r3", "rand_r4"]
    assert isinstance(criterion, torch.nn.CrossEntropyLoss) or isinstance(criterion, torch.nn.BCELoss)

    sampling_dist = torch.distributions.normal.Normal(0, epsilon)
    perturbation = sampling_dist.sample(batch.shape).to(device)
    if regularizer_type == "rand_r4":
        perturbation *= batch_masks
    perturbed_batch = batch + perturbation

    # Get the input gradient for the perturbed batch
    perturbed_batch.requires_grad = True
    model.zero_grad()
    y_hat = model(perturbed_batch)
    loss = criterion(y_hat.squeeze(), labels)
    loss.backward()
    input_grad = perturbed_batch.grad.data

    weight_sum = torch.tensor(0).to(device, dtype=torch.float32).requires_grad_()
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            weight_sum = weight_sum + torch.sum(module.weight ** 2)
    match regularizer_type:
        case "smooth_r3":
            # return the average input gradient
            return torch.mean(torch.abs(input_grad * batch_masks)) + weight_reg_coeff * weight_sum
        case "rand_r4":
            # return the maximum input gradient of only the masked regions
            samples_max_input_grad_elem_wise = torch.max(input_grad, dim=0).values
            non_zero_masks = (batch_masks > 0).sum().item()
            return torch.sum(torch.abs(samples_max_input_grad_elem_wise * batch_masks)) / non_zero_masks  + weight_reg_coeff * weight_sum

def input_gradient_pgd_regularizer(
    batch: torch.Tensor,
    labels: torch.Tensor,
    model: torch.nn.Module,
    batch_masks: torch.Tensor,
    criterion: torch.nn.Module,
    epsilon: float,
    num_iterations: int = 10,
    regularizer_type: str = "std",
    device: str = "cuda:0",
    weight_reg_coeff: float = 0.0,
    clip_grad_bound: float = None,
) -> torch.Tensor:
    """This function is used to compute the adversarial perturbation budget for the input data batch and return it to be used as a regularization term
    in order to optimize the behaviour of a model to ignore irrelevant features. This is not a certification technique, but does provide a minimal level
    of robustness to the model to perturbations of the unsalient features.

    Args:
        batch (torch.Tensor): The input data batch, should have shape [batchsize x ... ].
        labels (torch.Tensor): The labels for the input data batch, should have shape [batchsize x output_dim] if cross entropy,
            [batchsize] if binary cross entropy.
        model (torch.nn.Module): The neural network model.
        batch_masks (torch.Tensor): The salient image masks for the input data batch, should match the shape of the input.
        loss_fn (str): The loss function to use. Supported: "binary_cross_entropy", "cross_entropy".
        epsilon (float): The adversarial perturbation budget.
        num_iterations (int, optional): Number of PGD iterations. Defaults to 10. If the number of iterations is 1, the attack is FGSM
        regularizer_type (str, optional): The type of regularizer to use. Can be "grad_cert", "r4", "r3", "std" or "pgd". Defaults to "std".
        device (_type_, optional): The device to use for computation. Defaults to "cuda:0".

    """
    assert regularizer_type in ["pgd_r4", "pgd_ex+r3", "std", "pgd_ex", "r3"]
    assert batch_masks is not None
    #! =========================== Checkpoint to make training faster ===========================
    if regularizer_type == "std":
        return 0
    weight_sum = torch.tensor(0).to(device, dtype=torch.float32).requires_grad_()
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            weight_sum = weight_sum + torch.sum(module.weight ** 2)
    if regularizer_type == "r3":
        model.zero_grad()
        reg_term = torch.tensor(0).to(device, dtype=torch.float32).requires_grad_()
        batch = batch.requires_grad_()
        y_hat = model(batch)
        loss = criterion(y_hat.squeeze(), labels)
        loss.backward()
        reg_term = reg_term + torch.sum((batch.grad.data.reshape(batch_masks.shape) * batch_masks) ** 2)
        return weight_reg_coeff * weight_sum + reg_term
    #! ==================================== End of Checkpoint ===================================

    pgd_adv_input = batch
    perturbation_masks = batch_masks if regularizer_type in ["pgd_ex", "pgd_r4"] else torch.ones_like(batch_masks).to(device)
    for _ in range(num_iterations) :
        pgd_adv_input.requires_grad = True
        # We need it because otherwise the gradients will be accumulated with previous iterations
        #@ This also means we need to perform the regularization BEFORE the normal training step
        model.zero_grad()

        y_hat = model(pgd_adv_input)
        # We need to squeeze the logits when the loss is BCELoss because labels has size [batchsize] and y_hat [batchsize x 1]
        loss = criterion(y_hat.squeeze(), labels)
        loss.backward()

        adv_batch_step = pgd_adv_input + epsilon * perturbation_masks * torch.sign(pgd_adv_input.grad.data)
        delta = torch.clamp(adv_batch_step - batch, min=-epsilon, max=epsilon)
        adv_batch_step = torch.clamp(batch + delta, min=batch.min(), max=batch.max()).detach_()
        pgd_adv_input = adv_batch_step

    # Compute the loss for the interval regularizer
    match regularizer_type:
        case "pgd_ex":
            return criterion(model(pgd_adv_input), labels) + weight_reg_coeff * weight_sum
        case "pgd_r4":
            # One last time, we do a full forward and backward pass to get the input gradient for the pgd adversarial example
            pgd_adv_input.requires_grad = True
            model.zero_grad()
            y_hat = model(pgd_adv_input)
            # We squeeze here for the same reason as above
            loss = criterion(y_hat.squeeze(), labels)
            loss.backward()
            pgd_grad_reg = torch.sum(torch.abs(torch.mul(pgd_adv_input.grad.data, batch_masks)))
            if clip_grad_bound is not None:
                pgd_grad_reg = torch.clamp(pgd_grad_reg, min=0, max=clip_grad_bound)

            return pgd_grad_reg + weight_reg_coeff * weight_sum
        case "pgd_ex+r3":
            # r3 term
            reg_term = torch.tensor(0).to(device, dtype=torch.float32).requires_grad_()
            batch = batch.requires_grad_()
            #TODO see if we need to zero_grad this (grad accumulation)
            # model.zero_grad()
            y_hat = model(batch)
            loss = criterion(y_hat.squeeze(), labels)
            loss.backward()
            reg_term = reg_term + torch.sum((batch.grad.data.reshape(batch_masks.shape) * batch_masks) ** 2)
            # combine
            return weight_reg_coeff * weight_sum + criterion(model(pgd_adv_input), labels) + reg_term

# This only works for robust explanation constraints
def parameter_gradient_interval_regularizer(
    model: torch.nn.Sequential,
    batch: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: str,
    epsilon: float,
    model_epsilon: float,
    return_grads: bool = False,
) -> float | list[torch.Tensor]:
    """
    Compute an interval over the gradients of the loss with respect to the parameters to the network. Then compute the
    sum of the norms of the parameter gradient intervals and return it to be used as a regularization term. This can be
    used for robust (but un-certified) pre-training.

    Args:
        model (torch.nn.Sequential): The neural network model.
        batch (torch.Tensor): The input data batch, should have shape [batchsize x ... ].
        labels (torch.Tensor): The labels for the input data batch, should have shape [batchsize].
        loss_fn (str): Only "binary_cross_entropy" supported for now.
        epsilon (float): The training time input perturbation budget.
        model_epsilon (float): The training time model perturbation budget.
        return_grads (bool): Whether to return the gradients directly, instead of the regularization term.

    Returns:
        float | list[torch.Tensor]: The regularization term or the gradient lower bounds.
    """
    # we only support binary cross entropy loss for now
    if loss_fn != "binary_cross_entropy" and loss_fn != "cross_entropy":
        raise ValueError(f"Unsupported loss function: {loss_fn}")
    if not isinstance(model[-1], torch.nn.Sigmoid) and not isinstance(model[-1], torch.nn.Softmax):
        raise ValueError(f"Expected last layer to be sigmoid for binary cross entropy loss and softmax for cross entropy, got {model[-1]}")
    # remove the first module (copy of model) and the last module (sigmoid)
    modules = list(model.modules())[1:-1]

    # propagate through the forward pass
    intermediate = [(batch - epsilon, batch + epsilon)]
    for module in modules:
        intermediate.append(propagate_module_forward(module, *intermediate[-1], model_epsilon))
        interval_arithmetic.validate_interval(*intermediate[-1], msg=f"forward pass {module}")

    # propagate through the loss function
    logits_l, logits_u = intermediate[-1]
    logits_l, logits_u = logits_l.squeeze(-1), logits_u.squeeze(-1)
    dl_l, dl_u = model[-1](logits_l) - labels, model[-1](logits_u) - labels
    interval_arithmetic.validate_interval(dl_l, dl_u, msg="loss gradient")

    # propagate through the backward pass
    grads_l, grads_u = [], []
    for i in range(len(modules) - 1, -1, -1):
        # compute the gradients wrt the parameters
        gl, gu = compute_module_parameter_gradients(modules[i], dl_l, dl_u, *intermediate[i])
        grads_l.extend(gl)
        grads_u.extend(gu)
        # backpropagate the gradient through the module
        dl_l, dl_u = propagate_module_backward(modules[i], dl_l, dl_u, *intermediate[i], model_epsilon)
        interval_arithmetic.validate_interval(dl_l, dl_u, msg=f"backward pass {modules[i]}")

    # return the gradients or the regularization term
    if return_grads:
        grads_l.reverse()
        return grads_l

    norms = [torch.norm(u - l, p=2) for l, u in zip(grads_l, grads_u)]
    return sum(norms) / sum(g.nelement() for g in grads_l)


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
        dl_l (torch.Tensor): Lower bound on the gradient of the loss wrt the output of this module.
        dl_u (torch.Tensor): Upper bound on the  gradient of the loss wrt the output of this module.
        x_l (torch.Tensor): Lower bound on the input to the module.
        x_u (torch.Tensor): Upper bound on the input to the module.
        model_epsilon (float): The model perturbation budget.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Lower and upper bound on the gradient of the loss wrt the module input.
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
        W, dilation, stride = module.weight, module.dilation, module.stride
        padding, groups = module.padding, module.groups
        W_l, W_u = W - model_epsilon, W + model_epsilon

        # function that computes the gradient of the convolution wrt the input
        def conv_grad(dl_, W_):
            return torch.nn.grad.conv2d_input(x_l.shape, W_, dl_, stride, padding, dilation, groups)

        dl_l, dl_u = interval_arithmetic.propagate_linear_transform(dl_l, dl_u, W_l, W_u, transform=conv_grad)
    elif isinstance(module, torch.nn.Flatten):
        dl_l, dl_u = torch.reshape(dl_l, x_l.size()), torch.reshape(dl_u, x_u.size())
    else:
        raise ValueError(f"Unsupported module type: {type(module)}")
    interval_arithmetic.validate_interval(dl_l, dl_u, msg=f"module {module}")
    return dl_l, dl_u


def compute_module_parameter_gradients(
    module: torch.nn.Module,
    dl_l: torch.Tensor,
    dl_u: torch.Tensor,
    x_l: torch.Tensor,
    x_u: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Compute an interval over the gradients of the loss with respect to the parameters of the given module.

    Args:
        module (torch.nn.Module): The module to propagate through.
        dl_l (torch.Tensor): Lower bound on the gradient of the loss wrt the output of this module.
        dl_u (torch.Tensor): Upper bound on the gradient gradient of the loss wrt the output of this module.
        x_l (torch.Tensor): Lower bound on the input to the module.
        x_u (torch.Tensor): Upper bound on the input to the module.

    Returns:
        tuple[list[torch.Tensor], list[torch.Tensor]]: List of tuples containing the lower and upper bounds on the
            gradients of the loss with respect to the parameters of the module.
    """
    interval_arithmetic.validate_interval(dl_l, dl_u, msg="dl input")
    # get the parameters of the module
    parameters = list(module.parameters())
    # lists to store the gradient bounds
    grads_l, grads_u = [], []
    # if the module has no parameters return None
    if not parameters:
        return grads_l, grads_u

    # otherwise, compute gradients for the supported modules
    if isinstance(module, torch.nn.Linear):
        # compute gradients wrt the weight matrix
        dl_dW_l, dl_dW_u = interval_arithmetic.propagate_matmul(dl_l.T, dl_u.T, x_l.squeeze(-1), x_u.squeeze(-1))

        # compute gradients wrt the bias vector
        dl_db_l, dl_db_u = torch.sum(dl_l, dim=0), torch.sum(dl_u, dim=0)

        # store the gradients
        grads_l.extend([dl_db_l, dl_dW_l])
        grads_u.extend([dl_db_u, dl_dW_u])
    elif isinstance(module, torch.nn.Conv2d):
        # define gradient transform function
        W, dilation, stride, padding = module.weight, module.dilation, module.stride, module.padding

        # define the function that computes the gradient of the convolution wrt the weight matrix
        # note that this function is linear and hence we can compute an interval over the
        # output using Rump's algorithm.
        def transform(x, dl):
            return torch.nn.functional.grad.conv2d_weight(
                x, W.size(), dl, stride=stride, padding=padding, dilation=dilation
            )

        # compute gradients
        dl_db_l, dl_db_u = torch.sum(dl_l, dim=(0, 2, 3)), torch.sum(dl_u, dim=(0, 2, 3))
        dl_dW_l, dl_dW_u = interval_arithmetic.propagate_linear_transform(x_l, x_u, dl_l, dl_u, transform)

        # store the gradients
        grads_l.extend([dl_db_l, dl_dW_l])
        grads_u.extend([dl_db_u, dl_dW_u])
    else:
        raise ValueError(f"Unsupported module type: {type(module)}")
    return grads_l, grads_u


# if __name__ == "__main__":
#     # test the gradient calculations vs torch autograd
#     import sys
#
#     sys.path.append("..")
#     from datasets.oct_mnist import \
#         get_dataloaders  # pylint: disable=import-error
#     from deepmind import DeepMindSmall
#
#     # define model, dataset and optimizer
#     device = "cuda:0"
#     torch.manual_seed(0)
#     test_model = DeepMindSmall(1, 1)
#     dataloader, _ = get_dataloaders(64)
#     criterion = torch.nn.BCELoss(reduction="sum")
#     test_model = test_model.to(device)
#
#     # get the input data and pass through the model and loss
#     test_batch, test_labels = next(iter(dataloader))
#     test_batch, test_labels = test_batch.to(device), test_labels.to(device)
#     test_batch.requires_grad = True
#     intermediates = [test_batch]
#     for layer in list(test_model.modules())[1:]:
#         intermediates.append(layer(intermediates[-1]))
#         intermediates[-1].retain_grad()
#     loss = criterion(intermediates[-1].squeeze().float(), test_labels.squeeze().float())
#     loss.backward()
#     autograds_intermediate = [inter.grad for inter in intermediates]
#     autograds_parameters = [param.grad for param in test_model.parameters()]
#
#     # ======================== test the input_gradient_interval_regularizer ========================
#
#     # pass the data through our gradient interval regularizer with zero epsilon, which should match the exact grads
#     custom_grads = input_gradient_interval_regularizer(
#         test_model, test_batch, test_labels, "binary_cross_entropy", 0.0, 0.0, return_grads=True
#     )
#
#     for j in range(len(custom_grads)):
#         print(f"Layer {j}: matching gradients = {torch.allclose(custom_grads[j], autograds_intermediate[j])}")
#
#     # gradient regularization term:
#     reg = input_gradient_interval_regularizer(test_model, test_batch, test_labels, "binary_cross_entropy", 0.1, 0.0)
#     print("Input gradient interval regularization term (input perturbation):", reg)
#     reg = input_gradient_interval_regularizer(test_model, test_batch, test_labels, "binary_cross_entropy", 0.0, 0.001)
#     print("Input gradient interval regularization term (model perturbation):", reg)
#
#     # ======================== test the parameter_gradient_interval_regularizer ========================
#
#     # pass the data through our gradient interval regularizer with zero epsilon, which should match the exact grads
#     custom_grads = parameter_gradient_interval_regularizer(
#         test_model, test_batch, test_labels, "binary_cross_entropy", 0.0, 0.0, return_grads=True
#     )
#
#     for j in range(len(custom_grads)):
#         print(f"Parameter {j}: matching gradients = {torch.allclose(custom_grads[j], autograds_parameters[j])}")
#
#     # gradient regularization term:
#     reg = parameter_gradient_interval_regularizer(test_model, test_batch, test_labels, "binary_cross_entropy", 0.1, 0.0)
#     print("Parameter gradient interval regularization term (input perturbation):", reg)
#     reg = parameter_gradient_interval_regularizer(
#         test_model, test_batch, test_labels, "binary_cross_entropy", 0.0, 0.001
#     )
#     print("Parameter gradient interval regularization term (model perturbation):", reg)
#