"""
Helper functions for adding functional constraints to a gurobi model.
"""

from typing import Tuple, Optional
import gurobipy as gp
import numpy as np


def add_relu_bigm(
    model: gp.Model, x: gp.MVar, l: np.ndarray, u: np.ndarray, relax: bool = False
) -> Tuple[gp.MVar, Optional[gp.MVar]]:
    """
    Add the constraints defining the function y = ReLU(x) to the gurobi model.
    If relax is True, then we'll use the triangle relaxation.
    Returns the MVar for y and optionally the binaries z.

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


def add_bilinear_matmul(
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


def add_heaviside(m: gp.Model, x: gp.MVar, l: np.ndarray, u: np.ndarray, relax_binaries: bool):
    """
    Add the term z = Heaviside(x) to the gurobi model. The Heaviside function is defined as
        z = 1 if x > 0
        z = 0 if x <= 0
    If relax_binaries is True, then we'll use the linear relaxation.

    Args:
        model (gp.Model): Gurobi model to add the constraints to.
        x (gp.MVar): [n x 1] Gurobi MVar for the input variable.
        l (np.ndarray): [n x 1] Array of lower bounds for the input variable x.
        u (np.ndarray): [n x 1] Array of upper bounds for the input variable x.
        relax_binaries (bool): If True, use the linear relaxation of the Heaviside function.

    Returns:
        z (gp.MVar): [n x 1] MVar for the output of the Heaviside function.
    """
    vtype = gp.GRB.CONTINUOUS if relax_binaries else gp.GRB.BINARY
    z = m.addMVar(shape=x.shape, lb=0, ub=1, vtype=vtype)
    m.addConstr(x <= z * u)
    m.addConstr(x >= (1 - z) * l)
    m.addConstrs(z[i] == 1 for i in range(x.shape[0]) if l[i] > 0)
    m.addConstrs(z[i] == 0 for i in range(x.shape[0]) if u[i] <= 0)
    return z
