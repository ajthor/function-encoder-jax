from abc import ABC, abstractmethod

from functools import partial

from jax import jit, vmap, random, value_and_grad, tree_util
import jax.numpy as jnp

import equinox as eqx

import optax

# from function_encoder.model.base import BaseModel
from function_encoder.coefficients import CoefficientMethod


# class FunctionEncoder(ABC):

#     def __init__(
#         self,
#         basis_functions,
#         method: CoefficientMethod,
#         inner_product: callable,
#     ):
#         self.basis_functions = basis_functions
#         self.method = method
#         self.inner_product = inner_product

#     @jit
#     def compute_representation(self, X, y):
#         """Compute the representation of the function encoder."""

#         G = self.basis_functions.forward(X)
#         coefficients = self.method.compute_coefficients(G, y)

#         return coefficients, G

#     @jit
#     def forward(self, X, example_X, example_y):
#         """Forward pass."""

#         coefficients, _ = self.compute_representation(example_X, example_y)

#         G = self.basis_functions.forward(X)
#         y = jnp.einsum("kmd,k->md", G, coefficients)

#         return y

#     def _tree_flatten(self):
#         children = (self.basis_functions,)
#         aux_data = {"method": self.method, "inner_product": self.inner_product}
#         return (children, aux_data)

#     @classmethod
#     def _tree_unflatten(cls, aux_data, children):
#         return cls(*children, **aux_data)

#     @partial(jit, static_argnames=["optimizer"])
#     def update(fe, X, y, example_X, example_y, optimizer, opt_state):
#         """Update the function encoder."""

#         loss, grads = value_and_grad(loss_function)(fe, X, y, example_X, example_y)

#         updates, opt_state = optimizer.update(grads, opt_state)
#         fe = optax.apply_updates(fe, updates)

#         return fe, opt_state, loss


# def loss_function(fe, X, y, example_X, example_y):
#     """Compute the loss."""

#     y_pred = fe.forward(X, example_X, example_y)
#     prediction_error = y - y_pred
#     # prediction_loss = fe.inner_product(prediction_error, prediction_error).mean()
#     prediction_loss = jnp.mean(jnp.sum(prediction_error**2, axis=1))

#     total_loss = prediction_loss

#     return total_loss


# tree_util.register_pytree_node(
#     FunctionEncoder,
#     FunctionEncoder._tree_flatten,
#     FunctionEncoder._tree_unflatten,
# )

from typing import Callable
from jaxtyping import Array, PRNGKeyArray


def least_squares(G, y):
    """Compute the coefficients using least squares."""

    # Compute the matrix G^T F
    F = jnp.einsum("kmd,md->k", G, y)

    # Compute the Gram matrix K = G^T G
    K = jnp.einsum("kmd,lmd->kl", G, G)

    # Solve the linear system
    coefficients = jnp.linalg.solve(K, F)

    return coefficients, K


class FunctionEncoder(eqx.Module):
    basis_functions: eqx.nn.MLP
    coefficients_method: Callable

    def __init__(
        self,
        basis_size: int,
        coefficients_method: Callable = least_squares,
        *args,
        key: PRNGKeyArray,
        **kwargs
    ):

        # Initialize the basis functions
        keys = random.split(key, basis_size)
        make_mlp = lambda key: eqx.nn.MLP(*args, **kwargs, key=key)
        self.basis_functions = eqx.filter_vmap(make_mlp)(keys)

        self.coefficients_method = coefficients_method

    def __call__(self, X: Array, example_X: Array, example_y: Array):
        forward = eqx.filter_vmap(self.basis_functions, in_axes=(0, None))

        Gx = forward(example_X)

        coefficients = self.coefficients_method(Gx, example_y)
        G = forward(X)

        # Compute the predictions y = c^T G
        y = jnp.einsum("kd,k->d", G, coefficients)

        return y
