from typing import Callable, Mapping

from jax import random
import jax.numpy as jnp

import equinox as eqx
import optax

from jaxtyping import Array, PRNGKeyArray

from function_encoder.model.mlp import MLP
from function_encoder.function_encoder import FunctionEncoder

import tqdm


class EigenOperatorEncoder(eqx.Module):
    function_encoder: FunctionEncoder
    eigenvalues: Array

    def __init__(
        self,
        basis_size: int,
        *args,
        key: PRNGKeyArray,
        **kwargs,
    ):
        fe_key, eig_key = random.split(key, 2)

        self.function_encoder = FunctionEncoder(
            basis_size=basis_size, *args, key=fe_key, **kwargs
        )

        self.eigenvalues = random.uniform(eig_key, (basis_size,))

    def compute_coefficients(self, example_X: Array, example_y: Array):
        """Compute the coefficients of the basis functions for the given data."""
        coefficients = self.function_encoder.compute_coefficients(example_X, example_y)
        return coefficients * self.eigenvalues

    def __call__(self, X: Array, coefficients: Array):
        """Forward pass."""
        return self.function_encoder(X, coefficients)


def train_operator_encoder(
    model: eqx.Module,
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
