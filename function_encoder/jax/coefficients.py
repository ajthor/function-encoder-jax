from typing import Callable

import jax.numpy as jnp

import equinox as eqx
import lineax as lx

from jaxtyping import Array


def monte_carlo_integration(f: Array, g: Array, inner_product: Callable):
    """Compute the coefficients using Monte Carlo integration.

    Directly computes the inner product between each basis function and the target
    function without solving a linear system.

    Args:
        f: Array of shape (m, d) representing target function values at m points
        g: Array of shape (m, k, d) representing k basis functions evaluated at m points
        inner_product: Function that computes inner product between two sets of vectors

    Returns:
        coefficients: Array of shape (k,) containing the coefficients for each basis function
        G: None (Monte Carlo integration doesn't compute the Gram matrix)
    """
    F = eqx.filter_vmap(inner_product, in_axes=(1, None))(g, f)
    coefficients = F
    return coefficients, None


def least_squares(f: Array, g: Array, inner_product: Callable, reg: float = 1e-3):
    """Compute the coefficients using least squares.

    Solves the linear system G @ coefficients = F where:
    - G is the Gram matrix of inner products between basis functions
    - F is the vector of inner products between basis functions and target function

    Args:
        f: Array of shape (m, d) representing target function values at m points
        g: Array of shape (m, k, d) representing k basis functions evaluated at m points
        inner_product: Function that computes inner product between two sets of vectors
        reg: Regularization parameter added to diagonal of Gram matrix

    Returns:
        coefficients: Array of shape (k,) containing the coefficients for each basis function
        G: Array of shape (k, k) containing the Gram matrix of basis functions
    """
    F = eqx.filter_vmap(inner_product, in_axes=(1, None))(g, f)
    G = eqx.filter_vmap(
        eqx.filter_vmap(inner_product, in_axes=(1, None)), in_axes=(None, 1)
    )(g, g)
    G_reg = G.at[jnp.diag_indices_from(G)].add(reg)
    coefficients = jnp.linalg.solve(G_reg, F)
    return coefficients, G
