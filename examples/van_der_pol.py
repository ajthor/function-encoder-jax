from typing import Callable, Optional, Tuple, Union, Any
import jax

jax.config.update("jax_enable_x64", True)

from jax import random
import jax.numpy as jnp
from jaxtyping import Float, Scalar

import equinox as eqx
import optax

from datasets.van_der_pol import VanDerPolDataset, van_der_pol

from function_encoder.jax.model.mlp import MLP
from function_encoder.jax.model.neural_ode import NeuralODE, ODEFunc, rk4_step
from function_encoder.jax.function_encoder import BasisFunctions, FunctionEncoder

import matplotlib.pyplot as plt
import tqdm

# Load dataset
rng = random.PRNGKey(42)

dataset = VanDerPolDataset(n_points=1000, n_example_points=100, dt_range=(0.1, 0.1))
dataset_jit = eqx.filter_jit(dataset)


def dataloader(batch_size: int = 50, *, rng: random.PRNGKey):
    while True:
        rng, key = random.split(rng)
        keys = random.split(key, batch_size)
        batch = eqx.filter_vmap(lambda key: dataset_jit(key=key))(keys)
        yield batch


rng, dataset_key = random.split(rng)
dataloader_iter = dataloader(batch_size=50, rng=dataset_key)

# Create model


# Define a wrapper to create NeuralODE with ODEFunc
def make_neural_ode(layer_sizes, *, key, **kwargs):
    mlp = MLP(layer_sizes, key=key, **kwargs)
    ode_func = ODEFunc(mlp)
    return NeuralODE(ode_func, rk4_step)


rng, basis_key = random.split(rng)

n_basis = 10
basis_functions = BasisFunctions(
    basis_size=n_basis,
    basis_type=make_neural_ode,
    layer_sizes=[3, 64, 64, 2],
    key=basis_key,
)

model = FunctionEncoder(basis_functions)


# Train model


def compute_pred(model, X, coefficients):
    y_pred = eqx.filter_vmap(model, in_axes=(eqx.if_array(0), None))(X, coefficients)
    return y_pred


@eqx.filter_value_and_grad
def loss_function(model, batch):
    mu, y0, dt, y1, y0_example, dt_example, y1_example = batch
    coefficients, _ = eqx.filter_vmap(model.compute_coefficients)(
        (y0_example, dt_example), y1_example
    )
    y1_pred = eqx.filter_vmap(compute_pred, in_axes=(None, 0, 0))(
        model, (y0, dt), coefficients
    )
    pred_loss = optax.squared_error(y1, y1_pred).mean()
    return pred_loss


@eqx.filter_jit
@eqx.debug.assert_max_traces(max_traces=1)
def train_step(
    model: eqx.Module,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    batch: Any,
) -> Tuple[eqx.Module, optax.OptState, Float]:
    loss, grads = loss_function(model, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))
opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

num_epochs = 1000
with tqdm.tqdm(range(num_epochs)) as tqdm_bar:
    for epoch in tqdm_bar:
        batch = next(dataloader_iter)
        model, opt_state, loss = train_step(model, optimizer, opt_state, batch)

        if epoch % 10 == 0:
            tqdm_bar.set_postfix_str(f"Loss: {loss:.2e}")


# Plot a grid of evaluations

# Generate a batch of functions for plotting
plot_keys = random.split(rng, 9)
batch = jax.vmap(dataset)(plot_keys)

mu, y0, dt, y1, y0_example, dt_example, y1_example = batch

# Precompute the coefficients for the batch using vmap like in training
coefficients, G = eqx.filter_vmap(model.compute_coefficients)(
    (y0_example, dt_example), y1_example
)

fig, ax = plt.subplots(3, 3, figsize=(10, 10))

for i in range(3):
    for j in range(3):
        idx = i * 3 + j

        # Plot a single trajectory
        _mu = mu[idx]
        _y0 = random.uniform(
            random.PRNGKey(idx),
            (1, 2),
            minval=dataset.y0_range[0],
            maxval=dataset.y0_range[1],
        )
        # We use the coefficients that we computed before
        _c = coefficients[idx : idx + 1]  # Keep batch dimension
        s = 0.1  # Time step for simulation
        n = int(10 / s)
        _dt = jnp.array([s])

        # Integrate the true trajectory using scan
        def true_step(x, _):
            dx = rk4_step(van_der_pol, x, _dt[0], mu=_mu)
            x_next = x + dx
            return x_next, x_next
        
        _, y_true = jax.lax.scan(true_step, _y0.squeeze(), None, length=n)
        y = jnp.concatenate([_y0.squeeze()[None, :], y_true])  # Include initial state

        # Integrate the predicted trajectory using scan
        def pred_step(x, _):
            x_input = x[None, :]  # Add batch dimension for model call
            dt_input = _dt[None, :]  # Add batch dimension for dt
            x_next_delta = eqx.filter_vmap(model, in_axes=(eqx.if_array(0), None))(
                (x_input, dt_input), _c
            )
            x_next = x + x_next_delta.squeeze()
            return x_next, x_next
        
        _, pred_traj = jax.lax.scan(pred_step, _y0.squeeze(), None, length=n)
        pred = jnp.concatenate([_y0.squeeze()[None, :], pred_traj])  # Include initial state

        ax[i, j].set_xlim(-5, 5)
        ax[i, j].set_ylim(-5, 5)
        (_t,) = ax[i, j].plot(y[:, 0], y[:, 1], label="True")
        (_p,) = ax[i, j].plot(pred[:, 0], pred[:, 1], label="Predicted")

fig.legend(
    handles=[_t, _p],
    loc="outside upper center",
    bbox_to_anchor=(0.5, 0.95),
    ncol=2,
    frameon=False,
)

plt.show()

# Save the model (JAX equivalent - save pytree)
import pickle

with open("van_der_pol_model.pkl", "wb") as f:
    pickle.dump(model, f)
