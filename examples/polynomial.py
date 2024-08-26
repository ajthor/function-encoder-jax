import jax

jax.config.update("jax_enable_x64", True)

from jax import random
import jax.numpy as jnp

import equinox as eqx
import optax

from datasets import load_dataset

from function_encoder.losses import basis_normalization_loss
from function_encoder.function_encoder import FunctionEncoder, train_model

import matplotlib.pyplot as plt

# Load dataset

ds = load_dataset("ajthor/polynomial")
ds = ds.with_format("jax")


# Create model

rng = random.PRNGKey(0)
rng, key = random.split(rng)

model = FunctionEncoder(
    basis_size=8,
    layer_sizes=(1, 32, 1),
    activation_function=jax.nn.tanh,
    key=key,
)

# Train


def loss_function(model, point):
    coefficients = model.compute_coefficients(point["X"][:, None], point["y"][:, None])
    y_pred = eqx.filter_vmap(model, in_axes=(eqx.if_array(0), None))(
        point["X"][:, None], coefficients
    )
    pred_loss = optax.squared_error(point["y"][:, None], y_pred).mean()
    norm_loss = basis_normalization_loss(model.basis_functions, point["X"][:, None])
    return pred_loss + norm_loss


model = train_model(model, ds["train"], loss_function)


# Plot

point = ds["train"].take(1)[0]

X = point["X"][:, None]
y = point["y"][:, None]

idx = jnp.argsort(X, axis=0).flatten()
X = X[idx]
y = y[idx]

coefficients = model.compute_coefficients(X, y)
y_pred = eqx.filter_vmap(model, in_axes=(eqx.if_array(0), None))(X, coefficients)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(X, y, label="True")
ax.scatter(X, y, label="Data", color="red")

ax.plot(X, y_pred, label="Predicted")

plt.show()
