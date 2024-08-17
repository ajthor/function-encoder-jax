from typing import Callable

from functools import partial

import jax
from jax import random
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

import equinox as eqx

import optax

from datasets import Dataset


def monte_carlo_integration(G, y):
    return jnp.einsum("mkd,md->k", G, y) / G.shape[0]


def least_squares(G, y):
    F = jnp.einsum("mkd,md->k", G, y)
    K = jnp.einsum("mkd,mld->kl", G, G)
    return jnp.linalg.solve(K, F)


class CustomMLP(eqx.Module):
    layers: tuple[eqx.nn.Linear, ...]

    def __init__(self, key):
        key1, key2, key3 = random.split(key, 3)
        self.layers = tuple(
            [
                eqx.nn.Linear(1, 32, key=key1),
                eqx.nn.Linear(32, 32, key=key2),
                eqx.nn.Linear(32, 1, key=key3),
            ]
        )

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jnp.tanh(layer(x))
        return self.layers[-1](x)


class FunctionEncoder(eqx.Module, strict=True):
    basis_functions: eqx.Module

    def __init__(
        self,
        basis_size: int,
        *args,
        key: PRNGKeyArray,
        **kwargs,
    ):

        # Initialize the basis functions
        keys = random.split(key, basis_size)
        make_basis = eqx.filter_vmap(lambda key: eqx.nn.MLP(*args, key=key, **kwargs))
        self.basis_functions = make_basis(keys)

    def __call__(self, X: Array, example_X: Array, example_y: Array):

        @eqx.filter_vmap(in_axes=(eqx.if_array(0), None))
        def forward(basis_functions, X):
            return basis_functions(X)

        Gx = eqx.filter_vmap(forward, in_axes=(None, 0))(
            self.basis_functions, example_X
        )
        coefficients = least_squares(Gx, example_y)

        G = forward(self.basis_functions, X)

        # Compute the predictions y = c^T G
        y = jnp.einsum("kd,k->d", G, coefficients)

        return y


def train(
    model: FunctionEncoder,
    ds: Dataset,
    optimizer: optax.GradientTransformation = None,
    opt_state: optax.OptState = None,
    steps: int = 100,
    batch_size: int = 32,
):

    if optimizer is None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(1e-3),
        )
        # optimizer = optax.MultiSteps(
        #     optimizer, every_k_schedule=batch_size
        # )  # Gradient accumulation

    if opt_state is None:
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    def loss_function(model, X, y, example_X, example_y):
        y_pred = eqx.filter_vmap(model, in_axes=(0, None, None))(
            X, example_X, example_y
        )
        return jnp.mean(jnp.linalg.norm(y - y_pred, axis=1) ** 2)

    @eqx.filter_jit
    def update(flattened_model, X, y, example_X, example_y, flattened_opt_state):
        model = jax.tree_util.tree_unflatten(treedef_model, flattened_model)
        opt_state = jax.tree_util.tree_unflatten(treedef_opt_state, flattened_opt_state)

        loss, grads = eqx.filter_value_and_grad(loss_function)(
            model, X, y, example_X, example_y
        )
        updates, update_opt_state = optimizer.update(grads, opt_state)
        update_model = optax.apply_updates(model, updates)
        # learning_rate = 0.1
        # update_model = jax.tree_util.tree_map(
        #     lambda m, g: m - learning_rate * g, model, grads
        # )

        flattened_update_model = jax.tree_util.tree_leaves(update_model)
        flattened_update_opt_state = jax.tree_util.tree_leaves(update_opt_state)
        return model, opt_state, loss

    flattened_model, treedef_model = jax.tree_util.tree_flatten(model)
    flattened_opt_state, treedef_opt_state = jax.tree_util.tree_flatten(opt_state)

    for i, point in enumerate(ds):
        x, y = point["x"], point["y"]
        example_x, example_y = x, y
        flattened_model, flattened_opt_state, loss = update(
            flattened_model, x, y, example_x, example_y, flattened_opt_state
        )

        if i % 10 == 0:
            print(f"Loss: {loss}")

    model = jax.tree_util.tree_unflatten(treedef_model, flattened_model)
    return model
