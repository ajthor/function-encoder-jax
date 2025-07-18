from typing import Callable, Tuple

import jax
from jax import random
import jax.numpy as jnp
from jaxtyping import Array

import equinox as eqx


class MLP(eqx.Module):
    params: Tuple
    activation_function: Callable = jax.nn.relu

    def __init__(
        self,
        layer_sizes: Tuple[int, ...],
        *,
        activation_function: Callable = jax.nn.relu,
        key: random.PRNGKey,
    ):

        params = []

        # Initialize the parameters
        for n_in, n_out in zip(layer_sizes[:-2], layer_sizes[1:-1]):
            key, w_key, b_key = random.split(key, 3)
            C = jnp.sqrt(1 / n_in)
            w = random.uniform(w_key, (n_in, n_out), minval=-C, maxval=C)
            b = random.uniform(b_key, (n_out,), minval=-C, maxval=C)

            params.append((w, b))

        # Initialize the output layer
        key, w_key, b_key = random.split(key, 3)
        C = jnp.sqrt(1 / layer_sizes[-2])
        w = random.uniform(
            w_key, (layer_sizes[-2], layer_sizes[-1]), minval=-C, maxval=C
        )
        b = random.uniform(b_key, (layer_sizes[-1],), minval=-C, maxval=C)

        params.append((w, b))

        self.params = tuple(params)
        self.activation_function = activation_function

    def __call__(self, X: Array):
        """Forward pass."""

        for w, b in self.params[:-1]:
            y = jnp.dot(X, w) + b
            X = self.activation_function(y)

        w, b = self.params[-1]
        y = jnp.dot(X, w) + b

        return y
