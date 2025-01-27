from typing import Callable

import jax.numpy as jnp

import equinox as eqx
import lineax as lx

from jaxtyping import Array


def monte_carlo_integration(G: Array, y: Array, inner_product: Callable):
    """Compute the coefficients using Monte Carlo integration."""
    F = eqx.filter_vmap(inner_product, in_axes=(1, None))(G, y)

    coefficients = F
    return coefficients


def least_squares(G: Array, y: Array, inner_product: Callable, reg: float = 1e-9):
    """Compute the coefficients using least squares."""
    F = eqx.filter_vmap(inner_product, in_axes=(1, None))(G, y)
    K = eqx.filter_vmap(
        eqx.filter_vmap(inner_product, in_axes=(1, None)), in_axes=(None, 1)
    )(G, G)
    K = K.at[jnp.diag_indices_from(K)].add(reg)

    coefficients = jnp.linalg.solve(K, F)
    return coefficients
