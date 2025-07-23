from functools import partial
from typing import Callable, Tuple, Iterable, Any
import jax

jax.config.update("jax_enable_x64", True)

from jax import random
import jax.numpy as jnp
from jaxtyping import Float, Scalar

import equinox as eqx
import optax

from datasets.polynomial import PolynomialDataset, dataloader

from function_encoder.jax.losses import basis_normalization_loss
from function_encoder.jax.function_encoder import FunctionEncoder, BasisFunctions

import tqdm

import matplotlib.pyplot as plt

# Load dataset

rng = random.PRNGKey(42)

dataset = PolynomialDataset(n_points=100, n_example_points=10)
dataset_jit = eqx.filter_jit(dataset)


rng, dataset_key = random.split(rng)
dataloader_iter = iter(dataloader(dataset_jit, rng=dataset_key, batch_size=50))

# Create model

rng = random.PRNGKey(0)
rng, key = random.split(rng)

basis_functions = BasisFunctions(
    basis_size=8, layer_sizes=("scalar", 32, "scalar"), key=key
)
model = FunctionEncoder(basis_functions=basis_functions)

# Train


def compute_pred(model, X, coefficients):
    y_pred = eqx.filter_vmap(model, in_axes=(eqx.if_array(0), None))(X, coefficients)
    return y_pred


@eqx.filter_value_and_grad
def loss_function(model, batch):
    X, y, example_X, example_y = batch
    coefficients, G = eqx.filter_vmap(model.compute_coefficients, in_axes=(0, 0))(
        example_X, example_y
    )
    y_pred = eqx.filter_vmap(compute_pred, in_axes=(None, 0, 0))(model, X, coefficients)
    pred_loss = optax.squared_error(y, y_pred).mean()
    return pred_loss


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
with tqdm.tqdm(range(num_epochs)) as tqdm_bar:
    for epoch in tqdm_bar:
        batch = next(dataloader_iter)
        model, opt_state, loss = train_step(model, opt, opt_state, batch)

        if epoch % 10 == 0:
            tqdm_bar.set_postfix_str(f"Loss: {loss:.2e}")


# Plot

rng, key = random.split(rng)
point = dataset_jit(key)

X, y, example_X, example_y = point

idx = jnp.argsort(X, axis=0).flatten()
X = X[idx]
y = y[idx]

coefficients, _ = model.compute_coefficients(example_X, example_y)
y_pred = eqx.filter_vmap(model, in_axes=(eqx.if_array(0), None))(X, coefficients)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(X, y, label="True")
ax.plot(X, y_pred, label="Predicted")
ax.scatter(example_X, example_y, label="Data", color="red")
plt.legend()
plt.show()
