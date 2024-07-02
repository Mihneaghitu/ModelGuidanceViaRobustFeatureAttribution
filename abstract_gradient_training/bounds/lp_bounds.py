"""Bounding methods using the gurobi linear programming solver."""

import time
import logging

import gurobipy as gp
import numpy as np
import torch

from abstract_gradient_training.bounds import bound_utils


def bound_forward_pass(
    param_l: list[torch.Tensor], param_u: list[torch.Tensor], x0_l: torch.Tensor, x0_u: torch.Tensor, **kwargs
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Given bounds on the parameters of the neural network and an interval over the input, compute bounds on the logits
    and intermediate activations of the network using a linear programming formulation.

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
    param_l, param_u, x0_l, x0_u = bound_utils.validate_forward_bound_input(param_l, param_u, x0_l, x0_u)
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
        x_l = x0_l[i]
        x_u = x0_u[i]
        act_l, act_u = bound_forward_pass_helper(param_l, param_u, x_l, x_u)
        lower_bounds.append(act_l)
        upper_bounds.append(act_u)

    # log the timing statistics
    avg_time = (time.time() - start) / batchsize
    logging.info("Solved LP bounds for %d instances. Avg bound time %.2fs.", batchsize, avg_time)

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
) -> tuple[np.ndarray, np.ndarray]:
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
    """
    # define model and input variable
    model = bound_utils.init_gurobi_model()
    model.setParam("NonConvex", 2)
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

        # compute the pre-activation bounds for this layer
        h_l, h_u = bound_objective_vector(model, W @ h + b)
        activations_l.append(h_l)
        activations_u.append(h_u)

        # add next hidden variable
        h = add_relu_triangle_constr(model, h, W, b, h_l, h_u)

    return activations_l, activations_u


def bound_objective_vector(model: gp.Model, objective: gp.MVar | gp.MLinExpr) -> tuple[np.ndarray, np.ndarray]:
    """

    Given a gurobi model and a vector of objectives, compute the minimum and maximum value of the objective over the
    model.

    Args:
        model (gp.Model): Gurobi model
        objective (gp.MVar | gp.MLinExpr): Objective to minimize/maximize over, either a gurobi variable or expression.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of two numpy arrays, the first containing the minimum of each objective
                                       and the second containing the maximum value.
    """
    N = objective.size
    L, U = np.zeros((N, 1)), np.zeros((N, 1))
    for i in range(N):
        model.setObjective(objective[i], gp.GRB.MINIMIZE)
        model.reset()
        model.optimize()
        assert model.status == gp.GRB.OPTIMAL
        L[i] = model.objVal
        model.setObjective(objective[i], gp.GRB.MAXIMIZE)
        model.reset()
        model.optimize()
        assert model.status == gp.GRB.OPTIMAL
        U[i] = model.objVal
    return L, U


def add_relu_triangle_constr(model: gp.Model, x: gp.MVar, W: gp.MVar, b: gp.MVar, l: np.ndarray, u: np.ndarray):
    """
    Add the constraints defining the triangle relaxation of the function y = ReLU(Wx + b) to the gurobi model.
    Returns the MVar for y.

    Args:
        model (gp.Model): Gurobi model to add the constraints to.
        x (gp.MVar): [m x 1] Gurobi MVar for the input variable.
        W (gp.MVar): [n x m] Gurobi MVar for the weight matrix.
        b (gp.MVar): [n x 1] Gurobi MVar for the bias vector.
        l (np.ndarray): [n x 1] Array of lower bounds for the input variable x.
        u (np.ndarray): [n x 1] Array of upper bounds for the input variable x.
    Returns:
        y (gp.MVar): [n x 1] MVar for the output of the ReLU
    """
    # check input shape
    n, m = W.shape
    assert x.shape == (m, 1)
    assert b.shape == (n, 1)
    assert l.shape == (n, 1)
    assert u.shape == (n, 1)

    # Define output variable
    y = model.addMVar(b.shape, lb=np.maximum(l, 0), ub=np.maximum(u, 0))

    # Triangle constraints only valid when l <= 0 <= u
    l = np.minimum(l, 0)
    u = np.maximum(u, 0)

    # Add triangle constraints
    model.addConstr(y >= 0)
    model.addConstr(y >= W @ x + b)
    model.addConstr(y <= u * (W @ x + b - l) / (u - l))

    return y
