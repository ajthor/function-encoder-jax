from typing import Callable

from jax import random
import jax.numpy as jnp

import equinox as eqx
import optax

from jaxtyping import Float, Array, PRNGKeyArray

from function_encoder.model.mlp import MLP

import tqdm


def monte_carlo_integration(G: Array, y: Array):
    """Compute the coefficients using Monte Carlo integration."""
    F = jnp.einsum("kmd,md->k", G, y)
    return F / (y.shape[0] ** 2)


def least_squares(G: Array, y: Array):
    """Compute the coefficients using least squares."""
    F = jnp.einsum("kmd,md->k", G, y)
    K = jnp.einsum("kmd,lmd->kl", G, G)
    # K = K.at[jnp.diag_indices_from(K)].add(1 / y.shape[0] ** 2)
    coefficients = jnp.linalg.solve(K, F)
    return coefficients


class FunctionEncoder(eqx.Module):
    basis_functions: eqx.nn.MLP
    coefficients_method: Callable

    def __init__(
        self,
        basis_size: int,
        coefficients_method: Callable = least_squares,
        *args,
        key: PRNGKeyArray,
        **kwargs,
    ):

        # Initialize the basis functions
        keys = random.split(key, basis_size)
        make_mlp = lambda key: MLP(*args, **kwargs, key=key)
        self.basis_functions = eqx.filter_vmap(make_mlp)(keys)

        self.coefficients_method = coefficients_method

    def compute_coefficients(self, example_X: Array, example_y: Array):
        """Compute the coefficients of the basis functions for the given data."""
        forward = eqx.filter_vmap(
            lambda model, x: model(x), in_axes=(eqx.if_array(0), None)
        )
        G = forward(self.basis_functions, example_X)
        coefficients = self.coefficients_method(G, example_y)

        return coefficients

    def __call__(self, X: Array, coefficients: Array):
        """Compute the function approximation."""
        forward = eqx.filter_vmap(
            lambda model, x: model(x), in_axes=(eqx.if_array(0), None)
        )
        G = forward(self.basis_functions, X)
        y = jnp.einsum("kmd,k->md", G, coefficients)

        return y


def train_function_encoder(
    model: FunctionEncoder,
    ds,
    loss_function: Callable,
    learning_rate: float = 1e-3,
    gradient_accumulation_steps: int = 10,
):
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=learning_rate),
    )
    # Gradient accumulation
    opt = optax.MultiSteps(opt, every_k_schedule=gradient_accumulation_steps)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def update(model, point, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_function)(model, point)
        updates, opt_state = opt.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    with tqdm.tqdm(enumerate(ds), total=ds.num_rows) as tqdm_bar:
        for i, point in tqdm_bar:
            model, opt_state, loss = update(model, point, opt_state)

            if i % 10 == 0:
                tqdm_bar.set_postfix_str(f"Loss: {loss:.2e}")

    return model
