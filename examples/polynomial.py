import jax

jax.config.update("jax_enable_x64", True)

from jax import random
import jax.numpy as jnp

import equinox as eqx

from datasets import Dataset, load_dataset

import optax

from function_encoder.function_encoder import FunctionEncoder

import matplotlib.pyplot as plt


ds = load_dataset("ajthor/polynomial")
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


opt = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(1e-3),
)
opt = optax.MultiSteps(opt, every_k_schedule=10)  # Gradient accumulation
opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))


def loss_fn(model, X, y, example_X, example_y):
    coefficients = model.compute_coefficients(example_X, example_y)

    y_pred = model(X, coefficients)
    pred_error = y - y_pred
    pred_loss = jnp.mean(jnp.linalg.norm(pred_error, axis=-1) ** 2)

    return pred_loss


@eqx.filter_jit
def update(model, X, y, example_X, example_y, opt_state):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, X, y, example_X, example_y)

    updates, opt_state = opt.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss


for i, point in enumerate(ds["train"].take(1000)):
    X, y = (point["X"], point["y"])
    example_X, example_y = (X, y)
    model, opt_state, loss = update(model, X, y, example_X, example_y, opt_state)

    if i % 10 == 0:
        print(f"Loss: {loss}")


# Plot


point = ds["train"].take(1)[0]

X = point["X"]
y = point["y"]
example_X = X
example_y = y

idx = jnp.argsort(X, axis=0).flatten()
X = X[idx]
y = y[idx]

coefficients = model.compute_coefficients(example_X, example_y)
y_pred = model(X, coefficients)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(X, y, label="True")
ax.plot(X, y_pred, label="Predicted")

ax.scatter(example_X, example_y, color="red")

plt.show()
