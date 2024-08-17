from typing import Callable

import jax
from jax import random
import jax.numpy as jnp

from jaxtyping import Array

from function_encoder.model.base_model import BaseModel


@jax.tree_util.register_pytree_node_class
class MLP(BaseModel):

    def __init__(
        self,
        layer_sizes: Array,
        activation_function: Callable,
        key,
    ):

        params = []

        for n_in, n_out in zip(layer_sizes[:-2], layer_sizes[1:-1]):
            key, w_key, b_key = random.split(key, 3)

            C = jnp.sqrt(layer_sizes[0])

            w = random.uniform(w_key, (n_in, n_out), minval=-C, maxval=C)
            b = random.uniform(b_key, (n_out), minval=-C, maxval=C)

            params.append((w, b))

        key, w_key = random.split(key)
        w = random.uniform(
            w_key, (layer_sizes[-2], layer_sizes[-1]), minval=-C, maxval=C
        )

        params.append((w,))

        self.params = params
        self.activation_function = activation_function

    def __call__(self, X):

        for w, b in self.params[:-1]:
            X = jnp.dot(X, w) + b
            X = self.activation_function(X)

        (w,) = self.params[-1]
        X = jnp.dot(X, w)

        return X

    def tree_flatten(self):
        children = (self.params,)
        aux_data = {"activation_function": self.activation_function}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
