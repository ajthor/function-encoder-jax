import jax

jax.config.update("jax_enable_x64", True)

from jax import random
import jax.numpy as jnp

import equinox as eqx
import optax

from datasets import Dataset, load_dataset

from function_encoder.function_encoder import FunctionEncoder, train_function_encoder

import matplotlib.pyplot as plt


ds = load_dataset("ajthor/random_function")
ds = ds.with_format("jax")

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
    # Compute the forward pass.
    coefficients = model.compute_coefficients(point["X"], point["y"][:, None])
    y_pred = model(point["X"], coefficients)
    return optax.l2_loss(point["y"][:, None], y_pred).mean()


model = train_function_encoder(model, ds["train"].take(1000), loss_function)

# Plot


point = ds["train"].take(1)[0]

X = point["X"]
y = point["y"]
y = y[:, None]

idx = jnp.argsort(X, axis=0).flatten()
X = X[idx]
y = y[idx]

coefficients = model.compute_coefficients(X, y)
y_pred = model(X, coefficients)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(X, y, label="True")
ax.plot(X, y_pred, label="Predicted")

ax.scatter(X, y, color="red")

plt.show()
