from typing import Callable

from jax import random
import jax.numpy as jnp
from jaxtyping import Array, Key

import equinox as eqx


class MLP(eqx.Module):
    params: tuple
    activation_function: Callable = jnp.tanh

    def __init__(self, layer_sizes: tuple, activation_function: Callable, *, key: Key):

        params = []

        # Initialize the parameters
        for n_in, n_out in zip(layer_sizes[:-2], layer_sizes[1:-1]):
            key, w_key, b_key = random.split(key, 3)
            C = jnp.sqrt(1 / n_in)
            w = random.uniform(w_key, (n_in, n_out), minval=-C, maxval=C)
            b = random.uniform(b_key, (n_out,), minval=-C, maxval=C)

            params.append((w, b))

        # Initialize the output layer
        key, w_key = random.split(key)
        C = jnp.sqrt(1 / layer_sizes[-2])
        w = random.uniform(
            w_key, (layer_sizes[-2], layer_sizes[-1]), minval=-C, maxval=C
        )

        params.append((w,))

        self.params = tuple(params)
        self.activation_function = activation_function

    def __call__(self, X: Array):
        """Forward pass."""

        for w, b in self.params[:-1]:
            y = jnp.dot(X, w) + b
            X = self.activation_function(y)

        (w,) = self.params[-1]

        y = jnp.dot(X, w)

        return y
