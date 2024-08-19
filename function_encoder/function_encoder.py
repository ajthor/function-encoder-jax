from typing import Callable

from functools import partial

import jax
from jax import jit, vmap, random
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, Key

from dataclasses import dataclass, InitVar

import optax

from datasets import Dataset

from function_encoder.model.base import BaseModel
from function_encoder.model.mlp import MLP


def monte_carlo_integration(G, y):
    F = jnp.einsum("kmd,md->k", G, y)
    return F / y.shape[0]


def least_squares(G, y, reg=1e-3):
    F = jnp.einsum("kmd,md->k", G, y)
    K = jnp.einsum("kmd,lmd->kl", G, G)
    K = K.at[jnp.diag_indices_from(K)].add(reg)
    return jnp.linalg.solve(K, F)


@jax.tree_util.register_pytree_node_class
class FunctionEncoder(BaseModel):

    def __init__(
        self,
        basis_functions: tuple | None = None,
        coefficients_method: Callable = least_squares,
        basis_size: int = 11,
        *args,
        key: Key | None = None,
        **kwargs,
    ):

        if basis_functions is None and key is not None:
            # Initialize the basis functions.
            keys = random.split(key, basis_size)
            make_mlp = lambda key: MLP(*args, **kwargs, key=key)
            basis_functions = vmap(make_mlp, out_axes=0)(keys)

        self.basis_functions = basis_functions
        self.coefficients_method = coefficients_method
        # Internal forward function to vmap over the basis functions.
        self._forward = vmap(lambda model, x: model(x), in_axes=(0, None))

    def forward(self, X: Array):
        """Forward pass through the basis functions."""
        return self._forward(self.basis_functions, X)

    def compute_coefficients(self, example_X: Array, example_y: Array):
        """Compute the coefficients of the basis functions for the given data."""
        example_G = self.forward(example_X)
        coefficients = self.coefficients_method(example_G, example_y)
        return coefficients

    def __call__(self, X: Array, coefficients: Array):
        """Compute the function approximation."""
        G = self.forward(X)
        return jnp.einsum("kmd,k->md", G, coefficients)

    def tree_flatten(self):
        children = (self.basis_functions,)
        aux_data = {"coefficients_method": self.coefficients_method}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


def train(model, ds, batch_size=10, learning_rate=1e-3):

    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=learning_rate),
    )
    opt = optax.MultiSteps(opt, every_k_schedule=batch_size)  # Gradient accumulation
    opt_state = opt.init(model)

    def loss_fn(model, X, y, example_X, example_y):
        coefficients = model.compute_coefficients(example_X, example_y)
        y_pred = model(X, coefficients)
        return jnp.mean(jnp.linalg.norm(y - y_pred) ** 2)

    @jit
    def update(model, opt_state, X, y, example_X, example_y):
        loss, grads = jax.value_and_grad(loss_fn)(model, X, y, example_X, example_y)
        updates, opt_state = opt.update(grads, opt_state)
        model = optax.apply_updates(model, updates)
        return model, opt_state, loss

    for step, point in enumerate(ds):

        X, y, example_X, example_y = (
            point["X"],
            point["y"],
            point["example_X"],
            point["example_y"],
        )

        model, opt_state, loss = update(model, opt_state, X, y, example_X, example_y)

        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss}")

    return model
