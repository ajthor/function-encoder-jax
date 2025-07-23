"""Jittable polynomial dataset generation for JAX."""

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, PRNGKeyArray
from typing import Tuple
import equinox as eqx


def polyval(coefficients: Array, X: Array) -> Array:
    """Evaluate a polynomial at X using Horner's method.

    Args:
        coefficients: Polynomial coefficients of shape (degree + 1,)
        X: Input values of shape (n_points,)

    Returns:
        Polynomial values of shape (n_points,)
    """
    y = jnp.zeros_like(X)
    for c in coefficients:
        y = y * X + c
    return y


class PolynomialDataset(eqx.Module):
    """Equinox module for generating polynomial datasets.

    Args:
        coeff_range: Range for polynomial coefficients (min, max)
        n_points: Number of data points for training
        n_example_points: Number of example points for coefficient computation
        degree: Polynomial degree
    """

    coeff_range: Tuple[float, float]
    n_points: int
    n_example_points: int
    degree: int

    def __init__(
        self,
        coeff_range: Tuple[float, float] = (-1.0, 1.0),
        n_points: int = 1000,
        n_example_points: int = 100,
        degree: int = 3,
    ):
        self.coeff_range = coeff_range
        self.n_points = n_points
        self.n_example_points = n_example_points
        self.degree = degree

    def __call__(self, key: Array) -> Tuple[Array, Array, Array, Array]:
        """Generate a single polynomial dataset.

        Args:
            key: JAX random key

        Returns:
            Tuple of (X, y, example_X, example_y)
        """

        coefficients_key, x_key = random.split(key)

        # Generate polynomial coefficients
        coefficients = random.uniform(
            coefficients_key,
            (self.degree + 1,),
            minval=self.coeff_range[0],
            maxval=self.coeff_range[1],
        )

        # Sample random x values
        _X = random.uniform(
            x_key,
            (self.n_example_points + self.n_points,),
            minval=-1.0,
            maxval=1.0,
        )
        _y = polyval(coefficients, _X)

        # Split the data
        X = _X[self.n_example_points :]
        y = _y[self.n_example_points :]
        example_X = _X[: self.n_example_points]
        example_y = _y[: self.n_example_points]

        return X, y, example_X, example_y


def dataloader(dataset, rng: random.PRNGKey, *, batch_size: int):
    while True:
        rng, key = random.split(rng)
        keys = random.split(key, batch_size)
        batch = eqx.filter_vmap(lambda key: dataset(key=key))(keys)
        yield batch
