import jax

jax.config.update("jax_enable_x64", True)

from jax import random
import jax.numpy as jnp

from datasets import load_dataset

import equinox as eqx
import optax

from function_encoder.losses import basis_normalization_loss
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
    layer_sizes=(2, 32, 2),
    activation_function=jax.nn.tanh,
    key=model_key,
)


# Train

data = ds["train"].take(1)[0]
x = data["x"][:-1]
y = data["x"][1:]

G = eqx.filter_vmap(model.basis_functions)(x)
M = jnp.einsum("mkd,mld->kl", G, G)
L = jnp.linalg.cholesky(M)
rng, key = random.split(rng)
coefficients = jnp.dot(L, random.normal(key, (8, 10))).T

CG = eqx.filter_vmap(
    eqx.filter_vmap(model, in_axes=(eqx.if_array(0), None)),
    in_axes=(None, eqx.if_array(0)),
)

CGX = CG(x, coefficients)
CGY = CG(y, coefficients)

K = jnp.linalg.lstsq(CGX, CGY)[0]


# Plot

fig = plt.figure()
ax = fig.add_subplot(111)

# Plot the random functions.
for i in range(10):
    y = eqx.filter_vmap(model, in_axes=(eqx.if_array(0), None))(x, coefficients[:, i])
    ax.plot(x, y)


plt.show()
