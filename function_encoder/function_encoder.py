from abc import ABC, abstractmethod

from functools import partial

from jax import jit, vmap, random, value_and_grad, tree_util
import jax.numpy as jnp

import equinox as eqx


from typing import Callable
from jaxtyping import Array, PRNGKeyArray

from function_encoder.model.mlp import MLP


def monte_carlo_integration(G, y):
    """Compute the coefficients using Monte Carlo integration."""
    F = jnp.einsum("kmd,md->k", G, y)
    return F / G.shape[0]


def least_squares(G, y):
    """Compute the coefficients using least squares."""
    F = jnp.einsum("kmd,md->k", G, y)
    K = jnp.einsum("kmd,lmd->kl", G, G)
    coefficients = jnp.linalg.solve(K, F)
    return coefficients


class FunctionEncoder(eqx.Module):
    basis_functions: eqx.nn.MLP
    coefficients_method: Callable

    def __init__(
        self,
        basis_size: int,
        coefficients_method: Callable = least_squares,
        *args,
        key: PRNGKeyArray,
        **kwargs
    ):

        # Initialize the basis functions
        keys = random.split(key, basis_size)
        make_mlp = lambda key: MLP(*args, **kwargs, key=key)
        self.basis_functions = eqx.filter_vmap(make_mlp)(keys)

        self.coefficients_method = coefficients_method

    def compute_coefficients(self, example_X: Array, example_y: Array):
        """Compute the coefficients of the basis functions for the given data."""
        forward = eqx.filter_vmap(
            lambda model, x: model(x), in_axes=(eqx.if_array(0), None)
        )
        G = forward(self.basis_functions, example_X)
        coefficients = self.coefficients_method(G, example_y)

        return coefficients

    def __call__(self, X: Array, coefficients: Array):
        """Compute the function approximation."""
        forward = eqx.filter_vmap(
            lambda model, x: model(x), in_axes=(eqx.if_array(0), None)
        )
        G = forward(self.basis_functions, X)
        y = jnp.einsum("kmd,k->md", G, coefficients)

        return y
