import jax

jax.config.update("jax_enable_x64", True)

from jax import random
import jax.numpy as jnp

import equinox as eqx
import optax

from datasets import load_dataset

from function_encoder.jax.losses import basis_normalization_loss
from function_encoder.jax.function_encoder import FunctionEncoder, BasisFunctions
from function_encoder.jax.utils.training import train_step

import tqdm

import matplotlib.pyplot as plt

# Load dataset

ds = load_dataset("ajthor/polynomial")
ds = ds.with_format("jax")


# Create model

rng = random.PRNGKey(0)
rng, key = random.split(rng)

basis_functions = BasisFunctions(basis_size=8, layer_sizes=(1, 32, 1), key=key)
model = FunctionEncoder(basis_functions=basis_functions)

# Train


def loss_function(model, point):
    coefficients, G = model.compute_coefficients(
        point["X"][:, None], point["y"][:, None]
    )
    y_pred = eqx.filter_vmap(model, in_axes=(eqx.if_array(0), None))(
        point["X"][:, None], coefficients
    )
    pred_loss = optax.squared_error(point["y"][:, None], y_pred).mean()
    norm_loss = basis_normalization_loss(G)
    return pred_loss + norm_loss


# model = fit(model, ds["train"], loss_function)
opt = optax.MultiSteps(
    optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(1e-3),
    ),
    every_k_schedule=50
)
opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))


@eqx.filter_jit
def update(model, point, opt_state):
    return train_step(model, opt, opt_state, point, loss_function)


with tqdm.tqdm(ds["train"]) as tqdm_bar:
    for i, point in enumerate(tqdm_bar):
        model, opt_state, loss = update(model, point, opt_state)

        if i % 10 == 0:
            tqdm_bar.set_postfix_str(f"Loss: {loss:.2e}")


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

plt.show()
