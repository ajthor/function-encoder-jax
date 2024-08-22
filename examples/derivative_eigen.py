import jax

jax.config.update("jax_enable_x64", True)

from jax import random
import jax.numpy as jnp

import equinox as eqx
import optax

from datasets import load_dataset


from function_encoder.operator_encoder import (
    EigenOperatorEncoder,
    train_operator_encoder,
)

import matplotlib.pyplot as plt


ds = load_dataset("ajthor/derivative")
ds = ds.with_format("jax")

rng = random.PRNGKey(0)
rng, key = random.split(rng)

model = EigenOperatorEncoder(
    basis_size=8,
    layer_sizes=(1, 32, 32, 1),
    activation_function=jax.nn.tanh,
    key=key,
)


# Train


def loss_function(model, point):
    coefficients = model.compute_coefficients(point["X"], point["f"])
    Tf_pred = model(point["Y"], coefficients)
    return optax.l2_loss(point["Tf"], Tf_pred).mean()


model = train_operator_encoder(model, ds["train"], loss_function)

# Plot

point = ds["train"].take(1)[0]

coefficients = model.compute_coefficients(point["X"], point["f"])
Tf_pred = model(point["Y"], coefficients)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(point["Y"], point["Tf"], label="True")
ax.plot(point["Y"], Tf_pred, label="Predicted")

plt.show()
