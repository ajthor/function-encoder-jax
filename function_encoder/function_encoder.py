from typing import List

from abc import ABC, abstractmethod

from functools import partial

from jax import jit, vmap, tree_util
import jax.numpy as jnp

from function_encoder.model.base import BaseModel
from function_encoder.coefficients import CoefficientMethod


class FunctionEncoder(ABC):

    def __init__(
        self,
        basis_functions: List[BaseModel],
        method: CoefficientMethod,
        inner_product: callable,
    ):
        self.basis_functions = basis_functions
        self.method = method
        self.inner_product = inner_product

    @jit
    def compute_representation(self, X, y):
        """Compute the representation of the function encoder."""

        G = jnp.stack([g.forward(X) for g in self.basis_functions])
        coefficients = self.method.compute_coefficients(G, y)

        return coefficients, G

    @partial(vmap, in_axes=(None, 0, 0, 0))
    def forward(self, X, example_X, example_y):
        """Forward pass."""

        coefficients, _ = self.compute_representation(example_X, example_y)

        G = jnp.stack([g.forward(X) for g in self.basis_functions])
        y = jnp.einsum("kmd,k->md", G, coefficients)

        return y

    def loss(self, X, y, example_X, example_y):
        """Compute the loss."""

        y_pred = self.forward(X, example_X, example_y)
        prediction_error = y - y_pred
        prediction_loss = self.inner_product(prediction_error, prediction_error).mean()

        total_loss = prediction_loss

        return total_loss
    
    @partial(jit, static_argnames=["opt"])
    def update(self, X, y, example_X, example_y, optimizer, opt_state):
        """Update the function encoder."""
        
        loss, grads = jax.value_and_grad(self.loss, has_aux=True)(self, X, y, example_X, example_y)

        updates, opt_state = optimizer.update(grads, opt_state)
        self = optax.apply_updates(self, updates)

        return self, opt_state, loss


    def _tree_flatten(self):
        children = (self.basis_functions,)
        aux_data = {"method": self.method, "inner_product": self.inner_product}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

tree_util.register_pytree_node(
    FunctionEncoder,
    FunctionEncoder._tree_flatten,
    FunctionEncoder._tree_unflatten,
)