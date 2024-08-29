from typing import Callable

from jax import random
import jax.numpy as jnp

import equinox as eqx
import lineax as lx

from jaxtyping import Array, PRNGKeyArray

from function_encoder.model.mlp import MLP


def monte_carlo_integration(G: Array, y: Array):
    """Compute the coefficients using Monte Carlo integration."""
    F = jnp.einsum("mkd,md->k", G, y)
    return F / (y.shape[0] ** 2)


def least_squares(G: Array, y: Array, reg: float = 1e-9):
    """Compute the coefficients using least squares."""
    F = jnp.einsum("mkd,md->k", G, y) / y.shape[0]
    K = jnp.einsum("mkd,mld->kl", G, G) / y.shape[0]
    K = K.at[jnp.diag_indices_from(K)].add(reg)

    coefficients = jnp.linalg.solve(K, F)
    return coefficients

    # operator = lx.MatrixLinearOperator(K)
    # coefficients_solution = lx.linear_solve(operator, F)

    # return coefficients_solution.value


class BasisFunctions(eqx.Module):
    basis_functions: eqx.Module

    def __init__(
        self,
        basis_size: int,
        *args,
        basis_type: type = MLP,
        key: PRNGKeyArray,
        **kwargs,
    ):
        keys = random.split(key, basis_size)
        make_basis_function = lambda key: basis_type(*args, **kwargs, key=key)
        self.basis_functions = eqx.filter_vmap(make_basis_function)(keys)

    def __call__(self, X):
        """Compute the forward pass of the basis functions."""
        return eqx.filter_vmap(
            lambda model, x: model(x), in_axes=(eqx.if_array(0), None)
        )(self.basis_functions, X)


class FunctionEncoder(eqx.Module):
    basis_functions: BasisFunctions
    coefficients_method: Callable

    def __init__(
        self,
        *args,
        coefficients_method: Callable = least_squares,
        key: random.PRNGKey,
        **kwargs,
    ):
        self.basis_functions = BasisFunctions(*args, key=key, **kwargs)
        self.coefficients_method = coefficients_method

    def compute_coefficients(self, example_X: Array, example_y: Array):
        """Compute the coefficients of the basis functions for the given data."""
        G = eqx.filter_vmap(self.basis_functions)(example_X)
        coefficients = self.coefficients_method(G, example_y)

        return coefficients

    def __call__(self, X: Array, coefficients: Array):
        """Compute the function approximation."""
        G = self.basis_functions(X)
        y = jnp.einsum("kd,k->d", G, coefficients)

        return y


class ResidualFunctionEncoder(FunctionEncoder):
    average_function: MLP

    def __init__(
        self,
        basis_size: int,
        *args,
        basis_type: type = MLP,
        coefficients_method: Callable = least_squares,
        key: random.PRNGKey,
        **kwargs,
    ):
        fe_key, avg_key = random.split(key)
        super().__init__(
            *args,
            basis_size=basis_size,
            basis_type=basis_type,
            coefficients_method=coefficients_method,
            key=fe_key,
            **kwargs,
        )

        self.average_function = basis_type(*args, **kwargs, key=avg_key)

    def compute_coefficients(self, example_X: Array, example_y: Array):
        """Compute the coefficients of the basis functions for the given data."""
        avg = eqx.filter_vmap(self.average_function)(example_X)
        coefficients = super().compute_coefficients(example_X, example_y - avg)

        return coefficients

    def __call__(self, X: Array, coefficients: Array):
        """Compute the function approximation."""
        avg = self.average_function(X)
        y = super().__call__(X, coefficients)

        return y + avg
