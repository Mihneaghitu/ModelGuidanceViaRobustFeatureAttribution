"""
Formulate the bounding problem as an optimization problem and solve it using Gurobi.
"""

import time
import logging
from typing import Optional

import gurobipy as gp
import numpy as np
import torch

from abstract_gradient_training.bounds import bound_utils
from abstract_gradient_training.bounds import gurobi_utils
from abstract_gradient_training.bounds import mip_formulations
from abstract_gradient_training import interval_arithmetic


LOGGER = logging.getLogger(__name__)


def bound_forward_pass(
    param_l: list[torch.Tensor],
    param_u: list[torch.Tensor],
    x0_l: torch.Tensor,
    x0_u: torch.Tensor,
    relax_binaries: bool = False,
    relax_bilinear: bool = False,
    gurobi_kwargs: Optional[dict] = None,
    **kwargs,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Given bounds on the parameters of the neural network and an interval over the input, compute bounds on the logits
    and intermediate activations of the network using the following formulations:

        - MIQP: The exact formulation
        - QCQP: Relax binary variables to continuous
        - MILP: Relax bilinear constraints to linear envelopes
        - LP: Relax both binary and bilinear constraints

    Args:
        param_l (list[torch.Tensor]): list of lower bounds on the parameters of the network [W1, b1, ..., Wm, bm]
        param_u (list[torch.Tensor]): list of upper bounds on the parameters of the network [W1, b1, ..., Wm, bm]
        x0_l (torch.Tensor): [batchsize x input_dim x 1] Lower bound on the input to the network.
        x0_u (torch.Tensor): [batchsize x input_dim x 1] Upper bound on the input to the network.
        relax_binaries (bool): Whether to relax binary to continuous variables in the formulation.
        relax_bilinear (bool): Whether to relax bilinear to linear constraints in the formulation.
        gurobi_kwargs (Optional[dict]): Parameters to pass to the gurobi model.

    Returns:
        activations_l (list[torch.Tensor]): list of lower bounds on all (pre-relu) activations [x0, ..., xL] including
                                            the input and the logits. Each tensor xi has shape [batchsize x dim x 1].
        activations_u (list[torch.Tensor]): list of upper bounds on all (pre-relu) activations [x0, ..., xL] including
                                            the input and the logits. Each tensor xi has shape [batchsize x dim x 1].
    """
    # validate the input
    param_l, param_u, x0_l, x0_u = bound_utils.validate_forward_bound_input(param_l, param_u, x0_l, x0_u)
    device = x0_l.device

    # get the name of the bounding method
    if relax_binaries and relax_bilinear:
        method = "LP"
    elif relax_binaries:
        method = "QCQP"
    elif relax_bilinear:
        method = "MILP"
    else:
        method = "MIQP"

    # convert all inputs to numpy arrays
    param_l = [param.detach().cpu().numpy() for param in param_l]
    param_u = [param.detach().cpu().numpy() for param in param_u]
    x0_l = x0_l.detach().cpu().numpy()
    x0_u = x0_u.detach().cpu().numpy()

    # iterate over each instance in the batch
    batchsize = x0_l.shape[0]
    lower_bounds = []
    upper_bounds = []
    start = time.time()
    for i in range(batchsize):
        if i % (batchsize // 10 + 1) == 0:
            LOGGER.debug("Solved %s bounds for %d/%d instances.", method, i, batchsize)
        x_l = x0_l[i]
        x_u = x0_u[i]
        act_l, act_u, model = _bound_forward_pass_helper(
            param_l, param_u, x_l, x_u, relax_binaries, relax_bilinear, gurobi_kwargs
        )
        lower_bounds.append(act_l)
        upper_bounds.append(act_u)

    # log the timing statistics and final model information
    avg_time = (time.time() - start) / batchsize
    LOGGER.debug("Solved %s bounds for %d instances. Avg bound time %.2fs.", method, batchsize, avg_time)
    LOGGER.debug(gurobi_utils.get_gurobi_model_stats(model))

    # concatenate the results
    activations_l = [np.stack([act[i] for act in lower_bounds], axis=0) for i in range(len(lower_bounds[0]))]
    activations_u = [np.stack([act[i] for act in upper_bounds], axis=0) for i in range(len(upper_bounds[0]))]

    # convert the results back to torch tensors
    activations_l = [torch.tensor(act, device=device) for act in activations_l]
    activations_u = [torch.tensor(act, device=device) for act in activations_u]

    return activations_l, activations_u


def _bound_forward_pass_helper(
    param_l: list[np.ndarray],
    param_u: list[np.ndarray],
    x0_l: np.ndarray,
    x0_u: np.ndarray,
    relax_binaries: bool,
    relax_bilinear: bool,
    gurobi_kwargs: dict,
) -> tuple[np.ndarray, np.ndarray, gp.Model]:
    """
    Compute bounds on a single input by solving a mixed-integer program using gurobi.

    Args:
        param_l (list[np.ndarray]): list of lower bounds on the parameters of the network [W1, b1, ..., Wm, bm]
        param_u (list[np.ndarray]): list of upper bounds on the parameters of the network [W1, b1, ..., Wm, bm]
        x0_l (np.ndarray): [input_dim x 1] Lower bound on a single input to the network.
        x0_u (np.ndarray): [input_dim x 1] Upper bound on a single input to the network.
        relax_binaries (bool): Whether to relax binary to continuous variables in the formulation.
        relax_bilinear (bool): Whether to relax bilinear to linear constraints in the formulation.
        gurobi_kwargs (dict): Parameters to pass to the gurobi model.

    Returns:
        activations_l (list[np.ndarray]): list of lower bounds computed using bilinear programming on the (pre-relu)
                                            activations [x0, ..., xL] including the input and the logits.
        activations_u (list[np.ndarray]): list of upper bounds computed using bilinear programming on the (pre-relu)
                                            activations [x0, ..., xL] including the input and the logits.
        model (gp.Model): The gurobi model used to compute the bounds.
    """
    # define model and set the model parameters
    model = gurobi_utils.init_gurobi_model("Bounds")
    model.setParam("NonConvex", 2)
    if gurobi_kwargs is not None:
        for key, value in gurobi_kwargs.items():
            model.setParam(key, value)

    # add the input variable
    h = model.addMVar(x0_l.shape, lb=x0_l, ub=x0_u)
    n_layers = len(param_l) // 2

    activations_l = [x0_l]
    activations_u = [x0_u]

    # loop over each hidden layer
    for i in range(0, n_layers):
        # define MVar for the weights and biases
        W_l, W_u = param_l[2 * i], param_u[2 * i]
        b_l, b_u = param_l[2 * i + 1], param_u[2 * i + 1]
        W = model.addMVar(W_l.shape, lb=W_l, ub=W_u)
        b = model.addMVar(b_l.shape, lb=b_l, ub=b_u)

        # bounds on the input to the current layer:
        h_l, h_u = activations_l[-1], activations_u[-1]
        if i > 0:
            h_l, h_u = np.maximum(h_l, 0), np.maximum(h_u, 0)

        # add the bilinear term s = W @ h
        s = mip_formulations.add_bilinear_matmul(model, W, h, W_l, W_u, h_l, h_u, relax_bilinear)

        # first compute the pre-activation bounds for this layer using IBP
        h_l, h_u = numpy_to_torch_wrapper(interval_arithmetic.propagate_matmul_exact, W_l, W_u, h_l, h_u)
        h_l, h_u = h_l + b_l, h_u + b_u

        # if i == 0, the best we can do is ibp. otherwise, solve the min/max optimization problem for each neuron
        if i > 0:
            h_l_optimized, h_u_optimized = gurobi_utils.bound_objective_vector(model, s + b)
            if np.isinf(h_l_optimized).any() or np.isinf(h_u_optimized).any():
                LOGGER.debug(
                    "Inf in optimized bounds for layer %d, falling back to IBP. Consider increasing timeout.",
                    i,
                )
            h_l, h_u = np.maximum(h_l, h_l_optimized), np.minimum(h_u, h_u_optimized)

        # store the bounds
        activations_l.append(h_l)
        activations_u.append(h_u)

        # skip last layer
        if i == n_layers - 1:
            break

        # add next hidden variable
        h, _ = mip_formulations.add_relu_bigm(model, s + b, h_l, h_u, relax_binaries)

    return activations_l, activations_u, model


def bound_backward_pass(
    dL_min: torch.Tensor,
    dL_max: torch.Tensor,
    param_l: list[torch.Tensor],
    param_u: list[torch.Tensor],
    activations_l: list[torch.Tensor],
    activations_u: list[torch.Tensor],
    relax_binaries: bool = False,
    relax_bilinear: bool = False,
    gurobi_kwargs: Optional[dict] = None,
    **kwargs,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Given bounds on the parameters, intermediate activations and the first partial derivative of the loss, compute
    bounds on the gradients of the loss with respect to the parameters by solving an optimization problem.

    Args:
        dL_min (torch.Tensor): lower bound on the gradient of the loss with respect to the logits
        dL_max (torch.Tensor): upper bound on the gradient of the loss with respect to the logits
        param_l (list[torch.Tensor]): list of lower bounds on the parameters [W1, b1, ..., Wm, bm]
        param_u (list[torch.Tensor]): list of upper bounds on the parameters [W1, b1, ..., Wm, bm]
        activations_l (list[torch.Tensor]): list of lower bounds on the (pre-relu) activations [x0, ..., xL], including
                                            the input and the logits. Each tensor xi has shape [batchsize x n_i x 1].
        activations_u (list[torch.Tensor]): list of upper bounds on the (pre-relu) activations [x0, ..., xL], including
                                            the input and the logits. Each tensor xi has shape [batchsize x n_i x 1].
        relax_binaries (bool): Whether to relax binary to continuous variables in the formulation.
        relax_bilinear (bool): Whether to relax bilinear to linear constraints in the formulation.
        gurobi_kwargs (Optional[dict]): Parameters to pass to the gurobi model.

    Returns:
        grads_l (list[torch.Tensor]): list of lower bounds on the gradients given as a list [dW1, db1, ..., dWm, dbm]
        grads_u (list[torch.Tensor]): list of upper bounds on the gradients given as a list [dW1, db1, ..., dWm, dbm]
    """

    # validate the input
    dL_min, dL_max, param_l, param_u, activations_l, activations_u = bound_utils.validate_backward_bound_input(
        dL_min, dL_max, param_l, param_u, activations_l, activations_u
    )
    device = dL_min.device

    # get the name of the bounding method
    if relax_binaries and relax_bilinear:
        method = "LP"
    elif relax_binaries:
        method = "QCQP"
    elif relax_bilinear:
        method = "MILP"
    else:
        method = "MIQP"

    # convert all inputs to numpy arrays
    dL_min = dL_min.detach().cpu().numpy()
    dL_max = dL_max.detach().cpu().numpy()
    param_l = [param.detach().cpu().numpy() for param in param_l]
    param_u = [param.detach().cpu().numpy() for param in param_u]
    activations_l = [act.detach().cpu().numpy() for act in activations_l]
    activations_u = [act.detach().cpu().numpy() for act in activations_u]

    # get weight matrix bounds
    W_l, W_u = param_l[::2], param_u[::2]

    # iterate over each instance in the batch
    batchsize = activations_l[0].shape[0]
    lower_bounds = []
    upper_bounds = []
    start = time.time()
    for i in range(batchsize):
        if i % (batchsize // 10 + 1) == 0:
            LOGGER.debug("Solved %s backward pass bounds for %d/%d instances", method, i, batchsize)
        act_l = [act[i] for act in activations_l]
        act_u = [act[i] for act in activations_u]
        d_l = dL_min[i]
        d_u = dL_max[i]
        grads_l, grads_u, model = _bound_backward_pass_helper(
            d_l, d_u, W_l, W_u, act_l, act_u, relax_binaries, relax_bilinear, gurobi_kwargs
        )

        lower_bounds.append(grads_l)
        upper_bounds.append(grads_u)

    # log the timing statistics and final model information
    avg_time = (time.time() - start) / batchsize
    LOGGER.debug("Solved %s backward pass bounds for %d instances. Avg bound time %.2fs.", method, batchsize, avg_time)
    LOGGER.debug(gurobi_utils.get_gurobi_model_stats(model))

    # concatenate the results
    activations_l = [np.stack([act[i] for act in lower_bounds], axis=0) for i in range(len(lower_bounds[0]))]
    activations_u = [np.stack([act[i] for act in upper_bounds], axis=0) for i in range(len(upper_bounds[0]))]

    # convert the results back to torch tensors
    activations_l = [torch.tensor(act, device=device) for act in activations_l]
    activations_u = [torch.tensor(act, device=device) for act in activations_u]

    return activations_l, activations_u


def _bound_backward_pass_helper(
    dL_min: np.ndarray,
    dL_max: np.ndarray,
    W_l: list[np.ndarray],
    W_u: list[np.ndarray],
    activations_l: list[np.ndarray],
    activations_u: list[np.ndarray],
    relax_binaries: bool,
    relax_bilinear: bool,
    gurobi_kwargs: dict,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Compute backward pass bounds for a single input in the batch by formulating and solving an optimization problem
    for each gradient bound.

    Args:
        dL_min (np.ndarray): lower bound on the gradient of the loss with respect to the logits for a single input
        dL_max (np.ndarray): upper bound on the gradient of the loss with respect to the logits for a single input
        W_l (list[np.ndarray]): list of lower bounds on the weight matrices [W1, ..., Wm]
        W_u (list[np.ndarray]): list of upper bounds on the weight matrices [W1, ..., Wm]
        activations_l (list[torch.Tensor]): list of lower bounds on the (pre-relu) activations [x0, ..., xL], including
                                            the input and the logits. Each tensor has shape [n_i x 1].
        activations_u (list[torch.Tensor]): list of upper bounds on the (pre-relu) activations [x0, ..., xL], including
                                            the input and the logits. Each tensor has shape [n_i x 1].
        relax_binaries (bool): Whether to relax binary to continuous variables in the formulation.
        relax_bilinear (bool): Whether to relax bilinear to linear constraints in the formulation.
        gurobi_kwargs (dict): Parameters to pass to the gurobi model.
    Returns:
        grads_l (list[np.ndarray]): list of lower bounds on the gradients given as a list [dW1, db1, ..., dWm, dbm]
        grads_u (list[np.ndarray]): list of upper bounds on the gradients given as a list [dW1, db1, ..., dWm, dbm]
        model (gp.Model): The gurobi model used to compute the bounds.

    """

    # define model and set the model parameters
    model = gurobi_utils.init_gurobi_model("Bounds")
    model.setParam("NonConvex", 2)
    if gurobi_kwargs is not None:
        for key, value in gurobi_kwargs.items():
            model.setParam(key, value)

    if relax_bilinear:
        raise NotImplementedError("Relaxing bilinear constraints is not supported for the backward pass.")

    # compute the gradient of the loss with respect to the weights and biases of the last layer
    dW_min, dW_max = numpy_to_torch_wrapper(
        interval_arithmetic.propagate_matmul,
        dL_min,
        dL_max,
        np.maximum(activations_l[-2].T, 0),
        np.maximum(activations_u[-2].T, 0),
        interval_matmul="exact",
    )
    grads_l, grads_u = [dL_min, dW_min], [dL_max, dW_max]

    # compute gradients for each layer
    for i in range(len(W_l) - 1, 0, -1):
        # initialise variable for the weight matrix and current partial derivative
        dL = model.addMVar(shape=dL_min.shape, lb=dL_min, ub=dL_max)
        W = model.addMVar(shape=W_l[i].shape, lb=W_l[i], ub=W_u[i])
        # compute bounds on the next partial derivative
        dL_dz_min, dL_dz_max = gurobi_utils.bound_objective_vector(model, W.T @ dL)
        # initialise variables for the next partial derivative, activation and heaviside of the activation
        dL_dz = model.addMVar(shape=(W_l[i].shape[1], 1), lb=dL_dz_min, ub=dL_dz_max)
        act = model.addMVar(shape=activations_l[i].shape, lb=activations_l[i], ub=activations_u[i])
        # note that we define the heaviside function using the pre-activation bounds
        heavi = mip_formulations.add_heaviside(model, act, activations_l[i], activations_u[i], relax_binaries)
        # compute bounds on the next partial derivvative
        dL_min, dL_max = gurobi_utils.bound_objective_vector(model, dL_dz * heavi)
        # compute bounds on the partial derivative wrt the weights using ibp
        dW_min, dW_max = numpy_to_torch_wrapper(
            interval_arithmetic.propagate_matmul,
            dL_min,
            dL_max,
            np.maximum(activations_l[i - 1].T, 0) if i - 1 > 0 else activations_l[i - 1].T,
            np.maximum(activations_u[i - 1].T, 0) if i - 1 > 0 else activations_u[i - 1].T,
            interval_matmul="exact",
        )
        grads_l.append(dL_min)
        grads_l.append(dW_min)
        grads_u.append(dL_max)
        grads_u.append(dW_max)

    grads_l.reverse()
    grads_u.reverse()
    return grads_l, grads_u, model


def numpy_to_torch_wrapper(fn, *args, **kwargs):
    """
    Wrapper function to convert numpy arrays to torch tensors before calling the function.
    """
    ret = fn(*[torch.from_numpy(arg) for arg in args], **kwargs)
    return tuple(r.detach().cpu().numpy() for r in ret)
