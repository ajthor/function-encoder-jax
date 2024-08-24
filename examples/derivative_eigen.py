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

# Load dataset

ds = load_dataset("ajthor/derivative")
ds = ds.with_format("jax")


# Create model

rng = random.PRNGKey(0)
rng, key = random.split(rng)

model = EigenOperatorEncoder(
    basis_size=8,
    layer_sizes=(1, 64, 64, 1),
    activation_function=jax.nn.tanh,
    key=key,
)


# Train


def loss_function(model, point):
    coefficients = model.compute_coefficients(point["X"][:, None], point["f"][:, None])
    Tf_pred = eqx.filter_vmap(model, in_axes=(eqx.if_array(0), None))(
        point["Y"][:, None], coefficients
    )
    pred_loss = optax.l2_loss(point["Tf"][:, None], Tf_pred).mean()
    return pred_loss


model = train_operator_encoder(model, ds["train"], loss_function)


# Plot

point = ds["train"].take(1)[0]

X = point["X"][:, None]
f = point["f"][:, None]
Y = point["Y"][:, None]
Tf = point["Tf"][:, None]

coefficients = model.compute_coefficients(X, f)
Tf_pred = eqx.filter_vmap(model, in_axes=(eqx.if_array(0), None))(Y, coefficients)

fig = plt.figure()
ax = fig.add_subplot(111)

idx = jnp.argsort(X, axis=0).flatten()
X = X[idx]
f = f[idx]

idx = jnp.argsort(Y, axis=0).flatten()
Y = Y[idx]
Tf = Tf[idx]

ax.plot(X, f, label="Original")
ax.scatter(X, f, label="Data", color="red")

ax.plot(Y, Tf, label="True")
ax.plot(Y, Tf_pred, label="Predicted")

plt.show()
