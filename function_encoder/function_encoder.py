from abc import ABC, abstractmethod

from functools import partial

from jax import jit, vmap, value_and_grad, tree_util
import jax.numpy as jnp

import optax

from function_encoder.model.base import BaseModel
from function_encoder.coefficients import CoefficientMethod


class FunctionEncoder(ABC):

    def __init__(
        self,
        basis_functions: BaseModel,
        method: CoefficientMethod,
        inner_product: callable,
    ):
        self.basis_functions = basis_functions
        self.method = method
        self.inner_product = inner_product

    @jit
    def compute_representation(self, X, y):
        """Compute the representation of the function encoder."""

        G = self.basis_functions.forward(X)
        coefficients = self.method.compute_coefficients(G, y)

        return coefficients, G

    @jit
    def forward(self, X, example_X, example_y):
        """Forward pass."""

        coefficients, _ = self.compute_representation(example_X, example_y)

        G = self.basis_functions.forward(X)
        y = jnp.einsum("kmd,k->md", G, coefficients)

        return y

    def _tree_flatten(self):
        children = (self.basis_functions,)
        aux_data = {"method": self.method, "inner_product": self.inner_product}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    @partial(jit, static_argnames=["optimizer"])
    def update(fe, X, y, example_X, example_y, optimizer, opt_state):
        """Update the function encoder."""

        loss, grads = value_and_grad(loss_function)(fe, X, y, example_X, example_y)

        updates, opt_state = optimizer.update(grads, opt_state)
        fe = optax.apply_updates(fe, updates)

        return fe, opt_state, loss


def loss_function(fe, X, y, example_X, example_y):
    """Compute the loss."""

    y_pred = fe.forward(X, example_X, example_y)
    prediction_error = y - y_pred
    # prediction_loss = fe.inner_product(prediction_error, prediction_error).mean()
    prediction_loss = jnp.mean(jnp.sum(prediction_error**2, axis=1))

    total_loss = prediction_loss

    return total_loss


tree_util.register_pytree_node(
    FunctionEncoder,
    FunctionEncoder._tree_flatten,
    FunctionEncoder._tree_unflatten,
)
