from typing import Callable, Mapping

from jax import random
import jax.numpy as jnp

import equinox as eqx
import optax

from jaxtyping import Array, PRNGKeyArray

from function_encoder.model.mlp import MLP
from function_encoder.function_encoder import FunctionEncoder

import tqdm


class SVDOperatorEncoder(eqx.Module):
    source_encoder: FunctionEncoder
    target_encoder: FunctionEncoder
    singular_values: Array

    def __init__(
        self,
        basis_size: int,
        *args,
        key: PRNGKeyArray,
        **kwargs,
    ):
        source_key, sv = random.split(key)

        self.source_encoder = FunctionEncoder(
            basis_size=basis_size, *args, key=source_key, **kwargs
        )

        self.target_encoder = FunctionEncoder(
            basis_size=basis_size, *args, key=source_key, **kwargs
        )

        self.singular_values = random.uniform(sv, (basis_size,), minval=0, maxval=1)

    def compute_coefficients(self, example_X: Array, example_y: Array):
        """Compute the coefficients of the basis functions for the given data."""
        coefficients = self.source_encoder.compute_coefficients(example_X, example_y)
        return coefficients * jnp.abs(
            self.singular_values
        )  # Ensure positive singular values

    def __call__(self, X: Array, coefficients: Array):
        """Forward pass."""
        return jnp.real(self.target_encoder(X, coefficients))


class EigenOperatorEncoder(eqx.Module):
    function_encoder: FunctionEncoder
    eigenvalues: Array

    def __init__(
        self,
        basis_size: int,
        *args,
        key: PRNGKeyArray,
        **kwargs,
    ):
        fe_key, eig_key = random.split(key)

        self.function_encoder = FunctionEncoder(
            basis_size=basis_size, *args, key=fe_key, **kwargs
        )

        self.eigenvalues = random.uniform(eig_key, (basis_size,), minval=-1, maxval=1)

    def compute_coefficients(self, example_X: Array, example_y: Array):
        """Compute the coefficients of the basis functions for the given data."""
        coefficients = self.function_encoder.compute_coefficients(example_X, example_y)
        return coefficients * self.eigenvalues

    def __call__(self, X: Array, coefficients: Array):
        """Forward pass."""
        return self.function_encoder(X, coefficients)
