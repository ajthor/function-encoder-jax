import jax

jax.config.update("jax_enable_x64", True)

from jax import random
import jax.numpy as jnp

from datasets import load_dataset

import equinox as eqx
import optax

from function_encoder.function_encoder import FunctionEncoder, train_function_encoder

import matplotlib.pyplot as plt


ds = load_dataset("ajthor/derivative")
ds = ds.with_format("jax")

rng = random.PRNGKey(0)
source_key, target_key, operator_key = random.split(rng, 3)


source_encoder = FunctionEncoder(
    basis_size=8,
    layer_sizes=(1, 32, 1),
    activation_function=jax.nn.tanh,
    key=source_key,
)

target_encoder = FunctionEncoder(
    basis_size=8,
    layer_sizes=(1, 32, 1),
    activation_function=jax.nn.tanh,
    key=target_key,
)


# Train


# Train the source encoder.
def source_loss_function(model, point):
    coefficients = model.compute_coefficients(point["X"], point["f"])
    f_pred = model(point["X"], coefficients)
    return optax.l2_loss(point["f"], f_pred).mean()


source_encoder = train_function_encoder(
    source_encoder, ds["train"].take(1000), source_loss_function
)


# Train the target encoder.
def target_loss_function(model, point):
    coefficients = model.compute_coefficients(point["Y"], point["Tf"])
    Tf_pred = model(point["Y"], coefficients)
    return optax.l2_loss(point["Tf"], Tf_pred).mean()


target_encoder = train_function_encoder(
    target_encoder, ds["train"].take(1000), target_loss_function
)

# Train the operator.
ds_subset = ds["train"].take(1000)

source_coefficients = eqx.filter_vmap(source_encoder.compute_coefficients)(
    ds_subset["X"], ds_subset["f"]
)
target_coefficients = eqx.filter_vmap(target_encoder.compute_coefficients)(
    ds_subset["Y"], ds_subset["Tf"]
)

operator = jnp.linalg.lstsq(source_coefficients, target_coefficients)[0]


# Plot

point = ds["train"].take(1)[0]

source_coefficients = source_encoder.compute_coefficients(point["X"], point["f"])
target_coefficients = jnp.dot(source_coefficients, operator)
Tf_pred = target_encoder(point["Y"], target_coefficients)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(point["Y"], point["Tf"], label="True")
ax.plot(point["Y"], Tf_pred, label="Predicted")

plt.show()
