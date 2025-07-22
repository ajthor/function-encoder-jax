from typing import Callable, Tuple

import jax
from jax import random
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

import equinox as eqx


class MLP(eqx.Module):
    """A multi-layer perceptron neural network using Equinox Linear layers.
    
    This MLP uses Equinox's Linear layers which handle scalar inputs properly
    and maintain consistent shapes throughout the computation.
    
    Args:
        layer_sizes: Tuple of integers specifying the size of each layer,
                    including input and output dimensions
        activation_function: Activation function to apply between layers.
                           Defaults to jax.nn.relu
        key: JAX random key for parameter initialization
    """
    layers: Tuple[eqx.nn.Linear, ...]
    activation_function: Callable = jax.nn.relu

    def __init__(
        self,
        layer_sizes: Tuple[int, ...],
        *,
        activation_function: Callable = jax.nn.relu,
        key: PRNGKeyArray,
    ):
        keys = random.split(key, len(layer_sizes) - 1)
        layers = []
        
        # Create Linear layers for each consecutive pair
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layers.append(eqx.nn.Linear(n_in, n_out, key=keys[i]))
        
        self.layers = tuple(layers)
        self.activation_function = activation_function

    def __call__(self, x: Array) -> Array:
        """Forward pass through the network.
        
        Applies linear transformations followed by activation functions for all
        hidden layers, then applies the final linear transformation without
        activation for the output layer.
        
        Args:
            x: Input array (scalar or vector)
            
        Returns:
            Output array (scalar or vector)
        """
        # Apply all layers except the last with activation
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation_function(x)
        
        # Apply final layer without activation
        x = self.layers[-1](x)
        
        return x
