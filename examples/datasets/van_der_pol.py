"""Jittable van der pol dataset generation for JAX."""

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, PRNGKeyArray
from typing import Tuple
import equinox as eqx

from function_encoder.jax.model.neural_ode import rk4_step


def van_der_pol(t: Array, x: Array, mu: Array = 1.0) -> Array:
    """Van der Pol oscillator ODE.

    Args:
        t: Time (unused but needed for ODE interface)
        x: State vector of shape (..., 2) where x[..., 0] is position, x[..., 1] is velocity
        mu: Parameter controlling nonlinearity

    Returns:
        Time derivative dx/dt of shape (..., 2)
    """
    return jnp.stack(
        [x[..., 1], mu * (1 - x[..., 0] ** 2) * x[..., 1] - x[..., 0]], axis=-1
    )


class VanDerPolDataset(eqx.Module):
    """Equinox module for generating van der pol datasets.

    Args:
        n_points: Number of data points for training
        n_example_points: Number of example points for coefficient computation
        mu_range: Range for mu parameter (min, max)
        y0_range: Range for initial conditions (min, max)
        dt_range: Range for time steps (min, max)
    """

    n_points: int
    n_example_points: int
    mu_range: Tuple[float, float]
    y0_range: Tuple[float, float]
    dt_range: Tuple[float, float]

    def __init__(
        self,
        n_points: int = 1000,
        n_example_points: int = 100,
        mu_range: Tuple[float, float] = (0.5, 2.5),
        y0_range: Tuple[float, float] = (-3.5, 3.5),
        dt_range: Tuple[float, float] = (0.01, 0.1),
    ):
        self.n_points = n_points
        self.n_example_points = n_example_points
        self.mu_range = mu_range
        self.y0_range = y0_range
        self.dt_range = dt_range

    def __call__(
        self, key: Array
    ) -> Tuple[Array, Array, Array, Array, Array, Array, Array]:
        """Generate a single van der pol dataset.

        Args:
            key: JAX random key

        Returns:
            Tuple of (mu, y0, dt, y1, y0_example, dt_example, y1_example)
        """
        mu_key, y0_key, dt_key = random.split(key, 3)

        total_points = self.n_example_points + self.n_points

        # Generate a single mu parameter
        mu = random.uniform(
            mu_key, (), minval=self.mu_range[0], maxval=self.mu_range[1]
        )

        # Generate random initial conditions
        _y0 = random.uniform(
            y0_key, (total_points, 2), minval=self.y0_range[0], maxval=self.y0_range[1]
        )

        # Generate random time steps
        _dt = random.uniform(
            dt_key, (total_points,), minval=self.dt_range[0], maxval=self.dt_range[1]
        )

        # Integrate one step using RK4
        # rk4_step returns dx (the change), so we add it to y0 to get the next state y1
        _y1 = jax.vmap(lambda y0, dt: rk4_step(van_der_pol, y0, dt, mu=mu))(_y0, _dt)

        # Split the data
        y0_example = _y0[: self.n_example_points]
        dt_example = _dt[: self.n_example_points]
        y1_example = _y1[: self.n_example_points]

        y0 = _y0[self.n_example_points :]
        dt = _dt[self.n_example_points :]
        y1 = _y1[self.n_example_points :]

        return mu, y0, dt, y1, y0_example, dt_example, y1_example


def dataloader(dataset, rng: random.PRNGKey, *, batch_size: int):
    while True:
        rng, key = random.split(rng)
        keys = random.split(key, batch_size)
        batch = eqx.filter_vmap(lambda key: dataset(key=key))(keys)
        yield batch
