from typing import Callable

from jax import random

import equinox as eqx

from jaxtyping import Array, PRNGKeyArray

from function_encoder.jax.model.mlp import MLP
from function_encoder.jax.coefficients import least_squares
from function_encoder.jax.inner_products import standard_inner_product


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
    inner_product: Callable

    def __init__(
        self,
        basis_functions: BasisFunctions,
        coefficients_method: Callable = least_squares,
        inner_product: Callable = standard_inner_product,
    ):
        self.basis_functions = basis_functions
        self.coefficients_method = coefficients_method
        self.inner_product = inner_product

    def compute_coefficients(self, example_X: Array, example_y: Array):
        """Compute the coefficients of the basis functions for the given data."""
        g = eqx.filter_vmap(self.basis_functions)(example_X)
        coefficients, G = self.coefficients_method(g, example_y, self.inner_product)

        return coefficients, G

    def __call__(self, X: Array, coefficients: Array):
        """Compute the function approximation."""
        g = self.basis_functions(X)
        y = coefficients @ g

        return y


class ResidualFunctionEncoder(FunctionEncoder):
    average_function: MLP

    def __init__(
        self,
        basis_size: int,
        *args,
        basis_type: type = MLP,
        coefficients_method: Callable = least_squares,
        inner_product: Callable = standard_inner_product,
        key: random.PRNGKey,
        **kwargs,
    ):
        fe_key, avg_key = random.split(key)
        super().__init__(
            *args,
            basis_size=basis_size,
            basis_type=basis_type,
            coefficients_method=coefficients_method,
            inner_product=inner_product,
            key=fe_key,
            **kwargs,
        )

        self.average_function = basis_type(*args, **kwargs, key=avg_key)

    def compute_coefficients(self, example_X: Array, example_y: Array):
        """Compute the coefficients of the basis functions for the given data."""
        avg = eqx.filter_vmap(self.average_function)(example_X)
        coefficients, G = super().compute_coefficients(example_X, example_y - avg)

        return coefficients, G

    def __call__(self, X: Array, coefficients: Array):
        """Compute the function approximation."""
        avg = self.average_function(X)
        y = super().__call__(X, coefficients)

        return y + avg
