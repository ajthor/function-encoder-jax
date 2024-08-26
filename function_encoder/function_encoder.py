from typing import Callable

from jax import random
import jax.numpy as jnp

import equinox as eqx
import optax
import lineax as lx

from jaxtyping import Array, PRNGKeyArray

from function_encoder.model.mlp import MLP

import tqdm


def monte_carlo_integration(G: Array, y: Array):
    """Compute the coefficients using Monte Carlo integration."""
    F = jnp.einsum("mkd,md->k", G, y)
    return F / (y.shape[0] ** 2)


def least_squares(G: Array, y: Array, reg: float = 1e-9):
    """Compute the coefficients using least squares."""
    F = jnp.einsum("mkd,md->k", G, y) / y.shape[0]
    K = jnp.einsum("mkd,mld->kl", G, G) / y.shape[0]
    K = K.at[jnp.diag_indices_from(K)].add(reg)

    coefficients = jnp.linalg.solve(K, F)
    return coefficients

    # operator = lx.MatrixLinearOperator(K)
    # coefficients_solution = lx.linear_solve(operator, F)

    # return coefficients_solution.value


class BasisFunctions(eqx.Module):
    basis_functions: eqx.Module

    def __init__(
        self,
        basis_size: int,
        *args,
        basis_type: type = MLP,
        key: PRNGKeyArray,
        **kwargs,
    ):
        keys = random.split(key, basis_size)
        make_mlp = lambda key: basis_type(*args, **kwargs, key=key)
        self.basis_functions = eqx.filter_vmap(make_mlp)(keys)

    def __call__(self, X):
        """Compute the forward pass of the basis functions."""
        return eqx.filter_vmap(
            lambda model, x: model(x), in_axes=(eqx.if_array(0), None)
        )(self.basis_functions, X)


class FunctionEncoder(eqx.Module):
    basis_functions: BasisFunctions
    coefficients_method: Callable

    def __init__(
        self,
        coefficients_method: Callable = least_squares,
        *args,
        key: PRNGKeyArray,
        **kwargs,
    ):

        self.basis_functions = BasisFunctions(*args, key=key, **kwargs)
        self.coefficients_method = coefficients_method

    def compute_coefficients(self, example_X: Array, example_y: Array):
        """Compute the coefficients of the basis functions for the given data."""
        G = eqx.filter_vmap(self.basis_functions)(example_X)
        coefficients = self.coefficients_method(G, example_y)

        return coefficients

    def __call__(self, X: Array, coefficients: Array):
        """Compute the function approximation."""
        G = self.basis_functions(X)
        y = jnp.einsum("kd,k->d", G, coefficients)

        return y


def train_model(
    model: FunctionEncoder,
    ds,
    loss_function: Callable,
    learning_rate: float = 1e-3,
    gradient_accumulation_steps: int = 50,
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

    with tqdm.tqdm(enumerate(ds)) as tqdm_bar:
        for i, point in tqdm_bar:
            model, opt_state, loss = update(model, point, opt_state)

            if i % 10 == 0:
                tqdm_bar.set_postfix_str(f"Loss: {loss:.2e}")

    return model
