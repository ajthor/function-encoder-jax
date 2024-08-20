import jax

jax.config.update("jax_enable_x64", True)

from jax import random
import jax.numpy as jnp

import equinox as eqx

from datasets import Dataset

import optax

from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor

from function_encoder.function_encoder import FunctionEncoder

import matplotlib.pyplot as plt

rng = random.PRNGKey(0)

kernel = RBF(length_scale=0.2)
gp = GaussianProcessRegressor(kernel=kernel)


def generate_data(key):
    x_key, example_key = random.split(key)

    X = random.uniform(x_key, (200, 1), minval=-1, maxval=1)
    y = gp.sample_y(X[:, jnp.newaxis])

    idx = random.choice(example_key, X.shape[0], (100,), replace=False)

    X = X[~idx]
    y = y[~idx]

    example_X = X[idx]
    example_y = y[idx]

    return {"X": X, "y": y, "example_X": example_X, "example_y": example_y}


n_functions = 2000
rng, key = random.split(rng)
keys = random.split(key, n_functions)

data = jax.vmap(generate_data)(keys)

ds = Dataset.from_dict(data)
ds = ds.to_iterable_dataset()
ds = ds.with_format("jax")

rng, key = random.split(rng)

model = FunctionEncoder(
    basis_size=8, layer_sizes=(1, 32, 1), activation_function=jax.nn.tanh, key=key
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


for i, point in enumerate(ds):
    X, y, example_X, example_y = (
        point["X"],
        point["y"],
        point["example_X"],
        point["example_y"],
    )
    model, opt_state, loss = update(model, X, y, example_X, example_y, opt_state)

    if i % 10 == 0:
        print(f"Loss: {loss}")


# Plot


rng, coefficients_key, example_key = random.split(rng, 3)
C = random_polynomial(coefficients_key)

X = jnp.linspace(-1, 1, 1000).reshape(-1, 1)
y = jnp.polyval(C, X)

example_X = random.uniform(example_key, (10, 1), minval=-1, maxval=1)
example_y = jnp.polyval(C, example_X)

coefficients = model.compute_coefficients(example_X, example_y)
y_pred = model(X, coefficients)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(X, y, label="True")
ax.plot(X, y_pred, label="Predicted")

ax.scatter(example_X, example_y, color="red")

plt.show()
