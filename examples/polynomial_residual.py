import jax

jax.config.update("jax_enable_x64", True)

from jax import random
import jax.numpy as jnp

import equinox as eqx
import optax

from datasets import load_dataset

from function_encoder.jax.model.mlp import MLP
from function_encoder.jax.losses import basis_normalization_loss
from function_encoder.jax.function_encoder import ResidualFunctionEncoder
from function_encoder.jax.utils.training import fit

import matplotlib.pyplot as plt

# Load dataset

ds = load_dataset("ajthor/polynomial")
ds = ds.with_format("jax")


def add_bias(point):
    point["y"] = point["y"] + jnp.polyval(jnp.array([0, 2, 2]), point["X"])
    return point


# We add a bias to the dataset to demonstrate the residual method.
ds = ds.map(add_bias)

# Create model

key = random.PRNGKey(0)

model = ResidualFunctionEncoder(
    basis_size=8,
    layer_sizes=(1, 32, 1),
    activation_function=jax.nn.tanh,
    key=key,
)


# Train


def loss_function(model, point):
    res_pred = eqx.filter_vmap(model.average_function)(point["X"][:, None])
    res_loss = optax.squared_error(point["y"][:, None], res_pred).mean()

    coefficients, _ = model.compute_coefficients(
        point["X"][:, None], point["y"][:, None]
    )
    y_pred = eqx.filter_vmap(model, in_axes=(eqx.if_array(0), None))(
        point["X"][:, None], coefficients
    )
    pred_loss = optax.squared_error(point["y"][:, None], y_pred).mean()
    norm_loss = basis_normalization_loss(model.basis_functions, point["X"][:, None])
    return pred_loss + norm_loss + res_loss


model = fit(model, ds["train"], loss_function)


# Plot

point = ds["train"].take(1)[0]

X = point["X"][:, None]
y = point["y"][:, None]

idx = jnp.argsort(X, axis=0).flatten()
X = X[idx]
y = y[idx]

coefficients, _ = model.compute_coefficients(X, y)
y_pred = eqx.filter_vmap(model, in_axes=(eqx.if_array(0), None))(X, coefficients)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(X, y, label="True")
ax.scatter(X, y, label="Data", color="red")

ax.plot(X, y_pred, label="Predicted")

# Plot the bias.
avg_pred = eqx.filter_vmap(model.average_function)(X)
ax.plot(X, jnp.polyval(jnp.array([1, 2, 3]), X), label=f"Bias")
ax.plot(X, avg_pred, label=f"Avg function")

plt.legend()
plt.show()
