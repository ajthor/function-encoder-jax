from typing import Any, Tuple
import jax
from jax import random
import jax.numpy as jnp
from jaxtyping import Float

import equinox as eqx
import optax

from datasets.polynomial import PolynomialDataset, dataloader

from function_encoder.jax.model.mlp import MLP, MultiHeadMLP
from function_encoder.jax.function_encoder import FunctionEncoder, BasisFunctions
from function_encoder.jax.losses import basis_normalization_loss

import tqdm

# Load dataset

rng = random.PRNGKey(42)
dataset = PolynomialDataset(n_points=100, n_example_points=10)
dataset_jit = eqx.filter_jit(dataset)

rng, dataset_key = random.split(rng)
dataloader_iter = iter(dataloader(dataset_jit, rng=dataset_key, batch_size=50))

# Create model

rng = random.PRNGKey(0)
rng, basis_key, residual_key = random.split(rng, 3)

basis_functions = MultiHeadMLP(
    num_heads=8, layer_sizes=("scalar", 32, "scalar"), key=basis_key
)
residual_function = MLP(layer_sizes=("scalar", 32, "scalar"), key=residual_key)

model = FunctionEncoder(
    basis_functions=basis_functions, residual_function=residual_function
)

# Train


def compute_pred(model, X, coefficients):
    # Compute the prediction for each point in X using the coefficients
    y_pred = eqx.filter_vmap(model, in_axes=(eqx.if_array(0), None))(X, coefficients)
    return y_pred


@eqx.filter_value_and_grad
def loss_function(model, batch):
    X, y, example_X, example_y = batch

    # Compute the coefficients for each sample in the batch
    coefficients, G = eqx.filter_vmap(model.compute_coefficients, in_axes=(0, 0))(
        example_X, example_y
    )

    # Compute the prediction for each sample in the batch
    y_pred = eqx.filter_vmap(compute_pred, in_axes=(None, 0, 0))(model, X, coefficients)

    pred_loss = optax.squared_error(y, y_pred).mean()
    norm_loss = eqx.filter_vmap(basis_normalization_loss)(G).mean()

    # Compute residual loss (this is necessary to train the residual function)
    residual_pred = eqx.filter_vmap(eqx.filter_vmap(model.residual_function))(X)
    residual_loss = optax.squared_error(y, residual_pred).mean()

    return pred_loss + norm_loss + residual_loss


@eqx.filter_jit
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


opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))
opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

num_epochs = 1000
with tqdm.trange(num_epochs) as tqdm_bar:
    for epoch in tqdm_bar:
        batch = next(dataloader_iter)
        model, opt_state, loss = train_step(model, opt, opt_state, batch)
        tqdm_bar.set_postfix_str(f"Loss: {loss:.2e}")


# Plot

import matplotlib.pyplot as plt

rng, key = random.split(rng)
point = dataset_jit(key)

X, y, example_X, example_y = point

idx = jnp.argsort(X, axis=0).flatten()
X = X[idx]
y = y[idx]

coefficients, _ = model.compute_coefficients(example_X, example_y)
y_pred = eqx.filter_vmap(model, in_axes=(eqx.if_array(0), None))(X, coefficients)
res_pred = eqx.filter_vmap(model.residual_function)(X)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(X, y, label="True")
ax.plot(X, y_pred, label="Predicted")
ax.plot(X, res_pred, label="Residual", linestyle="--", color="black")
ax.scatter(example_X, example_y, label="Data", color="red")
plt.legend()
plt.show()
