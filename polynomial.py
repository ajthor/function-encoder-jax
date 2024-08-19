import jax
from jax import random, tree_util
import jax.numpy as jnp

import equinox as eqx

from datasets import Dataset

import optax

from function_encoder.function_encoder import FunctionEncoder

import matplotlib.pyplot as plt

rng = random.PRNGKey(0)


def random_polynomial(key, degree=3):
    coefficients = random.uniform(key, (degree + 1,), minval=-1, maxval=1)
    return coefficients


def data_generator():
    n_functions = 1000

    for i in range(n_functions):
        key, coefficients_key, x_key, example_key = random.split(rng, 4)

        coefficients = random_polynomial(coefficients_key)

        x = random.uniform(x_key, (100, 1), minval=-1, maxval=1)
        y = jnp.polyval(coefficients, x)

        example_x = random.uniform(example_key, (10, 1), minval=-1, maxval=1)
        example_y = jnp.polyval(coefficients, example_x)

        yield {"x": x, "y": y, "example_X": example_x, "example_y": example_y}


ds = Dataset.from_generator(data_generator)
ds = ds.to_iterable_dataset()
ds = ds.with_format("jax")


fe = FunctionEncoder(basis_size=8, layer_sizes=(1, 32, 1), key=rng)


opt = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(1e-3),
)
opt = optax.MultiSteps(opt, every_k_schedule=10)  # Gradient accumulation
opt_state = opt.init(eqx.filter(fe, eqx.is_inexact_array))


def loss_fn(fe, X, y, example_X, example_y):
    coefficients = fe.compute_coefficients(example_X, example_y)
    y_pred = fe(X, coefficients)

    pred_error = y - y_pred
    pred_loss = jnp.mean(jnp.linalg.norm(pred_error, axis=-1) ** 2)

    return pred_loss


@eqx.filter_jit
def update(fe, X, y, example_X, example_y, opt_state):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(fe, X, y, example_X, example_y)

    # If something breaks, i.e. nan, pass
    if not jnp.isfinite(loss):
        pass

    updates, opt_state = opt.update(grads, opt_state)
    fe = eqx.apply_updates(fe, updates)

    return fe, opt_state, loss


for i, point in enumerate(ds):
    X, y, example_X, example_y = (
        point["x"],
        point["y"],
        point["example_X"],
        point["example_y"],
    )
    fe, opt_state, loss = update(fe, X, y, example_X, example_y, opt_state)

    if i % 10 == 0:
        print(f"Loss: {loss}")


# Plot


rng, coefficients_key, example_key = random.split(rng, 3)
C = random_polynomial(coefficients_key)

x = jnp.linspace(-1, 1, 1000).reshape(-1, 1)
y = jnp.polyval(C, x)

example_x = random.uniform(example_key, (10, 1), minval=-1, maxval=1)
example_y = jnp.polyval(C, example_x)

coefficients = fe.compute_coefficients(example_x, example_y)
y_pred = fe(x, coefficients)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(x, y, label="True")
ax.plot(x, y_pred, label="Predicted")

ax.scatter(example_x, example_y, color="red")

plt.show()
