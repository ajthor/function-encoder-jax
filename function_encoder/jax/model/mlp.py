from typing import Callable, Tuple

import jax
from jax import random
import jax.numpy as jnp
from jaxtyping import Array

import equinox as eqx


class MLP(eqx.Module):
    """A multi-layer perceptron neural network implemented in JAX.
    
    This MLP uses uniform initialization with custom scaling and stores parameters
    as tuples of (weight, bias) pairs. The forward pass applies the activation
    function between all layers except the final output layer.
    
    Args:
        layer_sizes: Tuple of integers specifying the size of each layer,
                    including input and output dimensions
        activation_function: Activation function to apply between layers.
                           Defaults to jax.nn.relu
        key: JAX random key for parameter initialization
    """
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

        # Initialize the hidden layer parameters with uniform distribution
        # Scale factor C = sqrt(1/n_in) ensures gradients don't explode/vanish
        for n_in, n_out in zip(layer_sizes[:-2], layer_sizes[1:-1]):
            key, w_key, b_key = random.split(key, 3)
            C = jnp.sqrt(1 / n_in)
            w = random.uniform(w_key, (n_in, n_out), minval=-C, maxval=C)
            b = random.uniform(b_key, (n_out,), minval=-C, maxval=C)

            params.append((w, b))

        # Initialize the output layer parameters
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
        """Forward pass through the network.
        
        Applies linear transformations followed by activation functions for all
        hidden layers, then applies the final linear transformation without
        activation for the output layer.
        
        Args:
            X: Input array of shape (batch_size, input_dim) or (input_dim,)
            
        Returns:
            Output array of shape (batch_size, output_dim) or (output_dim,)
        """
        # Apply hidden layers with activation
        for w, b in self.params[:-1]:
            y = jnp.dot(X, w) + b
            X = self.activation_function(y)

        # Apply final layer without activation
        w, b = self.params[-1]
        y = jnp.dot(X, w) + b

        return y
