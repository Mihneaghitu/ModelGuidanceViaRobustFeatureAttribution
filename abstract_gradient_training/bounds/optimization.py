"""
Formulate the bounding problem as an optimization problem and solve it using Gurobi.
"""

import time
import logging
from typing import Optional, Tuple

import gurobipy as gp
import numpy as np
import torch

from abstract_gradient_training.bounds import input_validation
from abstract_gradient_training import interval_arithmetic


LOGGER = logging.getLogger(__name__)


def bound_forward_pass(
    param_l: list[torch.Tensor],
    param_u: list[torch.Tensor],
    x0_l: torch.Tensor,
    x0_u: torch.Tensor,
    relax_binaries: bool = False,
    relax_bilinear: bool = False,
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

    Returns:
        activations_l (list[torch.Tensor]): list of lower bounds on all (pre-relu) activations [x0, ..., xL] including
                                            the input and the logits. Each tensor xi has shape [batchsize x dim x 1].
        activations_u (list[torch.Tensor]): list of upper bounds on all (pre-relu) activations [x0, ..., xL] including
                                            the input and the logits. Each tensor xi has shape [batchsize x dim x 1].
    """
    # validate the input
    param_l, param_u, x0_l, x0_u = input_validation.validate_forward_bound_input(param_l, param_u, x0_l, x0_u)
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
        if i % (batchsize // 10) == 0:
            LOGGER.debug("Solved %s bounds for %d/%d instances.", method, i, batchsize)
        x_l = x0_l[i]
        x_u = x0_u[i]
        act_l, act_u, model = bound_forward_pass_helper(param_l, param_u, x_l, x_u, relax_binaries, relax_bilinear)
        lower_bounds.append(act_l)
        upper_bounds.append(act_u)

    # log the timing statistics and final model information
    avg_time = (time.time() - start) / batchsize
    LOGGER.debug("Solved %s bounds for %d instances. Avg bound time %.2fs.", method, batchsize, avg_time)
    LOGGER.debug(get_gurobi_model_stats(model))

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
    relax_binaries: bool,
    relax_bilinear: bool,
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
        name (str): Name to use for the gurobi model

    Returns:
        activations_l (list[np.ndarray]): list of lower bounds computed using bilinear programming on the (pre-relu)
                                            activations [x0, ..., xL] including the input and the logits.
        activations_u (list[np.ndarray]): list of upper bounds computed using bilinear programming on the (pre-relu)
                                            activations [x0, ..., xL] including the input and the logits.
    """
    # define model and input variable
    model = init_gurobi_model("Bounds", logfile="gurobi.log")
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

        # bounds on the input to the current layer:
        h_l, h_u = activations_l[-1], activations_u[-1]
        if i > 0:
            h_l, h_u = np.maximum(h_l, 0), np.maximum(h_u, 0)

        # add the bilinear term s = W @ h
        s = add_bilinear_term(model, W, h, W_l, W_u, h_l, h_u, relax_bilinear)

        # compute the pre-activation bounds for this layer
        if i == 0:
            # best we can do on the first layer is to use ibp
            h_l, h_u = numpy_to_torch_wrapper(interval_arithmetic.propagate_matmul_exact, W_l, W_u, h_l, h_u)
            h_l, h_u = h_l + b_l, h_u + b_u
        else:
            # otherwise solve the min/max optimization problem for each neuron in the layer
            h_l, h_u = bound_objective_vector(model, s + b)

        # store the bounds
        activations_l.append(h_l)
        activations_u.append(h_u)

        # skip last layer
        if i == n_layers - 1:
            break

        # add next hidden variable
        h, _ = add_relu_constr_bigm(model, s + b, h_l, h_u, relax_binaries)

    return activations_l, activations_u, model


def add_relu_constr_bigm(
    model: gp.Model, x: gp.MVar, l: np.ndarray, u: np.ndarray, relax: bool = False
) -> Tuple[gp.MVar, Optional[gp.MVar]]:
    """
    Add the constraints defining the triangle relaxation of the function y = ReLU(x) to the gurobi model.
    If relax is True, then we'll use the triangle relaxation. Returns the MVar for y and optionally the binaries z.

    Args:
        model (gp.Model): Gurobi model to add the constraints to.
        x (gp.MVar): [n x 1] Gurobi MVar for the input variable.
        l (np.ndarray): [n x 1] Array of lower bounds for the input variable x.
        u (np.ndarray): [n x 1] Array of upper bounds for the input variable x.
    Returns:
        y (gp.MVar): [n x 1] MVar for the output of the ReLU
        z (Optional[gp.MVar]): MVar for the activation set of the ReLU
    """
    # Check input shape
    assert x.shape == l.shape == u.shape

    # Define output variable
    y = model.addMVar(x.shape, lb=np.maximum(l, 0), ub=np.maximum(u, 0))

    # Add big-M constraints
    model.addConstr(y >= x)
    model.addConstr(y >= 0)
    model.addConstr(x <= u)
    model.addConstr(x >= l)

    # Use either the big-M or the triangle relaxation. Using the triangle relaxation directly is slightly faster than
    # using the big-M relaxation with continuous z variables.
    if relax:
        l = np.minimum(l, 0)
        u = np.maximum(u, 0)
        z = None
        model.addConstr(y <= u * (x - l) / (u - l))
    else:
        z = model.addMVar(x.shape, vtype=gp.GRB.BINARY)
        model.addConstr(y <= x - l * (1 - z))
        model.addConstr(y <= u * z)

    return y, z


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

    # use bilinear term
    if not relax:
        return W @ h

    # use linear envelope
    s = model.addMVar((m, 1), lb=-np.inf)
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
    return s


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


def numpy_to_torch_wrapper(fn, *args):
    """
    Wrapper function to convert numpy arrays to torch tensors before calling the function.
    """
    ret = fn(*[torch.from_numpy(arg) for arg in args])
    return tuple(r.detach().cpu().numpy() for r in ret)


def init_gurobi_model(name: str, quiet: bool = True, logfile: str = "") -> gp.Model:
    """
    Initialise a blank Gurobi model. Setting quiet = True will suppress all output from the model.
    """
    env = gp.Env(empty=True)
    env.setParam("LogToConsole", 0)
    env.start()
    m = gp.Model(name=name, env=env) if quiet else gp.Model(name=name)
    m.setParam("LogFile", logfile)
    return m


def get_gurobi_model_stats(model: gp.Model) -> str:
    """
    Return a string with statistics about the Gurobi model.

    Args:
        model (gp.Model):Gurobi model

    Returns:
        str: Model statistics in a human-readable format.
    """
    return (
        f"Statistics for model {model.ModelName}:\n"
        f"  {'Linear constraint matrix':<30}: {model.NumConstrs} Constrs, {model.NumVars} Vars, {model.NumNZs} NZs\n"
        f"  {'Quadratic constraints':<30}: {model.NumQConstrs} QConstrs, {model.NumQCNZs} QNZs\n"
        f"  {'SOS constraints':<30}: {model.NumSOS} SOS\n"
        f"  {'General constraints':<30}: {model.NumGenConstrs} GenConstrs\n"
        f"  {'Quadratic objective':<30}: {model.NumQNZs} NZs, {model.NumPWLObjVars} PWL \n"
        f"  {'Integer variables':<30}: {model.NumIntVars} IntVars, {model.NumBinVars} BinVars\n"
        f"  {'Solve time':<30}: {model.Runtime:.2f}s\n"
        f"  {'Status':<30}: {model.Status}\n"
    )
