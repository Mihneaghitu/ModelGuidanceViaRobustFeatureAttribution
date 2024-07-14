"""Compute bounds using linear programming."""

import time
import logging

import gurobipy as gp
import numpy as np
import torch

from abstract_gradient_training.bounds import input_validation
from abstract_gradient_training.bounds import gurobi_helpers


LOGGER = logging.getLogger(__name__)


def bound_forward_pass(
    param_l: list[torch.Tensor], param_u: list[torch.Tensor], x0_l: torch.Tensor, x0_u: torch.Tensor, **kwargs
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Given bounds on the parameters of the neural network and an interval over the input, compute bounds on the logits
    and intermediate activations of the network using the bilinear programming formulation.

    Args:
        param_l (list[torch.Tensor]): list of lower bounds on the parameters of the network [W1, b1, ..., Wm, bm]
        param_u (list[torch.Tensor]): list of upper bounds on the parameters of the network [W1, b1, ..., Wm, bm]
        x0_l (torch.Tensor): [batchsize x input_dim x 1] Lower bound on the input to the network.
        x0_u (torch.Tensor): [batchsize x input_dim x 1] Upper bound on the input to the network.

    Returns:
        activations_l (list[torch.Tensor]): list of lower bounds on all (pre-relu) activations [x0, ..., xL] including
                                            the input and the logits. Each tensor xi has shape [batchsize x dim x 1].
        activations_u (list[torch.Tensor]): list of upper bounds on all (pre-relu) activations [x0, ..., xL] including
                                            the input and the logits. Each tensor xi has shape [batchsize x dim x 1].
    """
    # validate the input
    param_l, param_u, x0_l, x0_u = input_validation.validate_forward_bound_input(param_l, param_u, x0_l, x0_u)
    device = x0_l.device

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
        if i % (batchsize // 10) == 0:
            LOGGER.debug("Solved LP bounds for %d/%d instances.", i, batchsize)
        x_l = x0_l[i]
        x_u = x0_u[i]
        act_l, act_u, model = bound_forward_pass_helper(param_l, param_u, x_l, x_u)
        lower_bounds.append(act_l)
        upper_bounds.append(act_u)

    # log the timing statistics and final model information
    avg_time = (time.time() - start) / batchsize
    LOGGER.debug("Solved LP bounds for %d instances. Avg bound time %.2fs.", batchsize, avg_time)
    LOGGER.debug(gurobi_helpers.get_gurobi_model_stats(model))

    # concatenate the results
    activations_l = [np.stack([act[i] for act in lower_bounds], axis=0) for i in range(len(lower_bounds[0]))]
    activations_u = [np.stack([act[i] for act in upper_bounds], axis=0) for i in range(len(upper_bounds[0]))]

    # convert the results back to torch tensors
    activations_l = [torch.tensor(act, device=device) for act in activations_l]
    activations_u = [torch.tensor(act, device=device) for act in activations_u]

    return activations_l, activations_u


def bound_forward_pass_helper(
    param_l: list[np.ndarray],
    param_u: list[np.ndarray],
    x0_l: np.ndarray,
    x0_u: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, gp.Model]:
    """
    Compute bounds on a single input by solving a linear program using gurobi.

    Args:
        param_l (list[np.ndarray]): list of lower bounds on the parameters of the network [W1, b1, ..., Wm, bm]
        param_u (list[np.ndarray]): list of upper bounds on the parameters of the network [W1, b1, ..., Wm, bm]
        x0_l (np.ndarray): [input_dim x 1] Lower bound on a single input to the network.
        x0_u (np.ndarray): [input_dim x 1] Upper bound on a single input to the network.

    Returns:
        activations_l (list[np.ndarray]): list of lower bounds computed using linear programming on the (pre-relu)
                                            activations [x0, ..., xL] including the input and the logits.
        activations_u (list[np.ndarray]): list of upper bounds computed using linear programming on the (pre-relu)
                                            activations [x0, ..., xL] including the input and the logits.
        model (gp.Model): Gurobi model used to compute the bounds.
    """
    # define model and input variable
    model = gurobi_helpers.init_gurobi_model("lp_bounds")
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

        s = add_bilinear_term(model, W, h, W_l, W_u, h_l, h_u, relax=True)

        # compute the pre-activation bounds for this layer
        h_l, h_u = gurobi_helpers.bound_objective_vector(model, s + b)
        activations_l.append(h_l)
        activations_u.append(h_u)

        # skip last layer
        if i == n_layers - 1:
            break

        # add next hidden variable
        h = add_relu_triangle_constr(model, s + b, h_l, h_u)

    return activations_l, activations_u, model


def add_bilinear_term(
    model: gp.Model,
    W: gp.MVar,
    h: gp.MVar,
    W_l: np.ndarray,
    W_u: np.ndarray,
    h_l: np.ndarray,
    h_u: np.ndarray,
    relax: bool = False,
) -> gp.MVar:
    """
    Add the bilinear term s = W @ h to the gurobi model. If relax is True, then the bilinear term is replaced with its
    linear envelope.

    Args:
        model (gp.Model): Gurobi model
        W (gp.MVar): [m x n] Gurobi MVar for the weight matrix
        h (gp.MVar): [n x 1] Gurobi MVar for the input vector
        W_l (np.ndarray): [m x n] Lower bounds on the weight matrix
        W_u (np.ndarray): [m x n] Upper bounds on the weight matrix
        h_l (np.ndarray): [n x 1] Lower bounds on the input vector
        h_u (np.ndarray): [n x 1] Upper bounds on the input vector
        relax (bool, optional): If True, use the linear envelope of the bilinear term. Defaults to False.

    Returns:
        gp.MVar: [m x 1] MVar representing the bilinear variable s.
    """
    # validate shapes of input
    m, n = W.shape
    assert W.shape == W_l.shape == W_u.shape
    assert h.shape == h_l.shape == h_u.shape == (n, 1)
    # declare output variable
    s = model.addMVar((m, 1), lb=-np.inf)
    if relax:  # use linear envelope
        # matrix of bilinear terms (W.T * h)
        S = model.addMVar(W.T.shape, lb=-np.inf, ub=np.inf)
        # lower bounds
        model.addConstr(S >= W_l.T * h + W.T * h_l - W_l.T * h_l)
        model.addConstr(S >= W_u.T * h + W.T * h_u - W_u.T * h_u)
        # upper bounds
        model.addConstr(S <= W_u.T * h + W.T * h_l - W_u.T * h_l)
        model.addConstr(S <= W.T * h_u + W_l.T * h - W_l.T * h_u)
        # sum along the rows to obtain the matrix - vector product
        model.addConstr(s == S.sum(0)[:, None])
    else:  # add the bilinear term
        model.addConstr(s == W @ h)

    return s


def add_relu_triangle_constr(model: gp.Model, x: gp.MVar, l: np.ndarray, u: np.ndarray):
    """
    Add the constraints defining the triangle relaxation of the function y = ReLU(x) to the gurobi model.
    Returns the MVar for y.

    Args:
        model (gp.Model): Gurobi model to add the constraints to.
        x (gp.MVar): [n x 1] Gurobi MVar for the input variable.
        l (np.ndarray): [n x 1] Array of lower bounds for the input variable x.
        u (np.ndarray): [n x 1] Array of upper bounds for the input variable x.
    Returns:
        y (gp.MVar): [n x 1] MVar for the output of the ReLU
    """
    # check input shapes
    assert x.shape == l.shape == u.shape

    # Define output variable
    y = model.addMVar(x.shape, lb=np.maximum(l, 0), ub=np.maximum(u, 0))

    # Triangle constraints only valid when l <= 0 <= u
    l = np.minimum(l, 0)
    u = np.maximum(u, 0)

    # Add triangle constraints
    model.addConstr(y >= 0)
    model.addConstr(y >= x)
    model.addConstr(y <= u * (x - l) / (u - l))

    return y
