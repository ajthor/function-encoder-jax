from collections.abc import Callable

from typing import Tuple, Optional, Union, Literal

import jax
import jax.numpy as jnp
from jax import random

import equinox as eqx
from equinox import field

from jaxtyping import Array


def symmetric_normalization(adjacency_matrix):
    # Add self-loops
    adjacency_matrix = adjacency_matrix + jnp.eye(adjacency_matrix.shape[0])
    # Compute the degree matrix
    degree_matrix = jnp.sum(adjacency_matrix, axis=1)
    # Compute D^{-1/2}
    inv_sqrt_degree = 1.0 / jnp.sqrt(degree_matrix)
    inv_sqrt_degree = jnp.diag(inv_sqrt_degree)
    # Normalize adjacency matrix: D^{-1/2} A D^{-1/2}
    adjacency_matrix_normalized = inv_sqrt_degree @ adjacency_matrix @ inv_sqrt_degree
    
    return adjacency_matrix_normalized

class GCNLayer(eqx.Module):
    weight: Array
    bias: Optional[Array]

    in_features: Union[int, Literal["scalar"]] = field(static=True)
    out_features: Union[int, Literal["scalar"]] = field(static=True)

    use_bias: bool = field(static=True)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        *,
        key: random.PRNGKey,
    ):
        w_key, b_key = random.split(key, 2)
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

        C = jnp.sqrt(1 / in_features)
        self.weights = random.uniform(w_key, (in_features, out_features), minval=-C, maxval=C)

        if use_bias:
            self.bias = random.uniform(b_key, (out_features,), minval=-C, maxval=C)
        else:
            self.bias = None

    def __call__(self, X: Array, adjacency_matrix: Array):
        """Forward pass."""
        X = jnp.dot(adjacency_matrix, X)
        y = jnp.dot(X, self.weights)

        if self.use_bias:
            y = y + self.bias

        return y
    

class GCN(eqx.Module):
    layers: Tuple[GCNLayer, ...]
    activation_function: Callable

    final_activation: bool = field(static=True)

    def __init__(
        self,
        layer_sizes: Tuple[int, ...],
        activation_function: Callable,
        use_bias: bool = True,
        final_activation: bool = True,
        *,
        key: random.PRNGKey,
    ):
        keys = random.split(key, len(layer_sizes) - 1)

        layers = []
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layer = GCNLayer(n_in, n_out, use_bias=use_bias, key=keys[i])
            layers.append(layer)

        self.layers = tuple(layers)
        self.activation_function = activation_function
        self.final_activation = final_activation
        
    def __call__(self, X: Array, adjacency_matrix: Array):
        """Forward pass."""
        for i, layer in enumerate(self.layers[:-1]):
            X = layer(X, adjacency_matrix)
            X = self.activation_function(X)

        X = self.layers[-1](X, adjacency_matrix)

        if self.final_activation:
            X = self.activation_function(X)

        return X