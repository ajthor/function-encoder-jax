from functools import partial

import jax

jax.config.update("jax_enable_x64", True)

from jax import random
import jax.numpy as jnp

import diffrax
import optax

from datasets import load_dataset

from function_encoder.model.neural_ode import NeuralODE
from function_encoder.function_encoder import FunctionEncoder, train_model

import matplotlib.pyplot as plt

# Load dataset

ds = load_dataset("ajthor/van_der_pol")
ds = ds.with_format("jax")


def shorten_traj(point, length):
    point["x"] = point["x"][:length]
    point["t"] = point["t"][:length]
    return point


# Create model

rng = random.PRNGKey(0)
rng, model_key = random.split(rng)

model = FunctionEncoder(
    basis_size=8,
    basis_type=NeuralODE,
    layer_sizes=(3, 64, 2),
    activation_function=jax.nn.tanh,
    key=model_key,
)

# model = NeuralODE(
#     layer_sizes=(3, 64, 2),
#     activation_function=jax.nn.tanh,
#     key=model_key,
# )


def predict_trajectory(f, y0, ts):
    """Computes a trajectory from an initial condition y0 at times ts."""

    def step_fn(y, t):
        y = f((y, t))
        return y, y

    return jax.lax.scan(step_fn, y0, ts)


# Train


def loss_function(model, point):
    t = point["t"].astype(jnp.float64)
    x = point["x"].astype(jnp.float64)
    ts = jnp.hstack([t[:-1, None], t[1:, None]])
    coefficients = model.compute_coefficients((x[:-1], ts), x[1:])
    y_pred = predict_trajectory(partial(model, coefficients=coefficients), x[0], ts)[1]
    pred_loss = optax.squared_error(x[1:], y_pred).mean()
    return pred_loss


model = train_model(
    model, ds["train"].take(100).map(partial(shorten_traj, length=10)), loss_function
)
model = train_model(
    model, ds["train"].take(100).map(partial(shorten_traj, length=100)), loss_function
)
model = train_model(model, ds["train"].take(50), loss_function)


# Plot

point = ds["train"].take(1).map(partial(shorten_traj, length=500))[0]
t = point["t"].astype(jnp.float64)
x = point["x"].astype(jnp.float64)

# y_pred = model((point["x"][0], point["t"]))
ts = jnp.hstack([t[:-1, None], t[1:, None]])
coefficients = model.compute_coefficients((x[:-1], ts), x[1:])
y_pred = predict_trajectory(partial(model, coefficients=coefficients), x[0], ts)[1]

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(point["x"][:, 0], point["x"][:, 1], label="True")
ax.plot(y_pred[:, 0], y_pred[:, 1], label="Predicted")

# Plot initial data
# ax.plot(
#     point["y"][:example_data_size, 0], point["y"][:example_data_size, 1], color="red"
# )

plt.show()
