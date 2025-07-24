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


class MultiHeadMLP(eqx.Module):
    """A multi-headed MLP that produces multiple outputs by reshaping a larger final layer.

    This is more efficient than using separate MLPs (like BasisFunctions) when you need
    multiple basis functions. Instead of multiple separate networks, it uses a single
    network with a final layer that's (num_heads * output_size) wide, then reshapes
    the output to (num_heads, output_size).

    Args:
        layer_sizes: Tuple of integers for hidden layer sizes (input size to last hidden)
        num_heads: Number of output heads (equivalent to basis_size in BasisFunctions)
        output_size: Size of each output head (typically 1 for scalar functions)
        activation_function: Activation function for hidden layers
        key: JAX random key for parameter initialization

    Example:
        # Create 8 basis functions, each producing scalar output
        multi_mlp = MultiHeadMLP(
            layer_sizes=(1, 32, 32),  # input=1, two hidden layers of 32
            num_heads=8,
            output_size=1,
            key=key
        )

        # Usage (equivalent to BasisFunctions)
        x = jnp.array(0.5)
        outputs = multi_mlp(x)  # Shape: (8,) - one scalar output per head
    """

    layers: Tuple[eqx.nn.Linear, ...]
    num_heads: int
    output_size: int
    activation_function: Callable

    def __init__(
        self,
        layer_sizes: Tuple[int, ...],
        num_heads: int,
        *,
        activation_function: Callable = jax.nn.relu,
        key: PRNGKeyArray,
    ):
        keys = random.split(key, len(layer_sizes) - 1)
        layers = []

        # Create hidden layers (all except the final output layer)
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-2], layer_sizes[1:-1])):
            layers.append(eqx.nn.Linear(n_in, n_out, key=keys[i]))

        # Create final layer with (num_heads * output_size) outputs
        output_size = layer_sizes[-1]
        final_output_size = (
            num_heads if output_size == "scalar" else num_heads * output_size
        )
        layers.append(eqx.nn.Linear(layer_sizes[-2], final_output_size, key=keys[-1]))

        self.layers = tuple(layers)
        self.num_heads = num_heads
        self.output_size = output_size
        self.activation_function = activation_function

    def __call__(self, x: Array) -> Array:
        """Forward pass producing multiple outputs.

        Args:
            x: Input array

        Returns:
            Array of shape (num_heads,) if output_size=1, or (num_heads, output_size)
        """
        # Forward pass through all layers except final
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation_function(x)

        # Final layer without activation
        x = self.layers[-1](x)

        if self.output_size == "scalar":
            # Reshape to (num_heads,) for scalar outputs
            x = jnp.reshape(x, (self.num_heads,))
        else:
            # Reshape to (num_heads, output_size) for vector outputs
            x = jnp.reshape(x, (self.num_heads, self.output_size))

        return x
