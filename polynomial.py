import jax

jax.config.update("jax_enable_x64", True)

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


def generate_data(key):
    c_key, x_key, example_key = random.split(key, 3)

    c = random_polynomial(c_key)

    x = random.uniform(x_key, (100, 1), minval=-1, maxval=1)
    y = jnp.polyval(c, x)

    example_x = random.uniform(example_key, (10, 1), minval=-1, maxval=1)
    example_y = jnp.polyval(c, example_x)

    return {"x": x, "y": y, "example_X": example_x, "example_y": example_y}


n_functions = 1000
rng, key = random.split(rng)
keys = random.split(key, n_functions)

# vmap
data = jax.vmap(generate_data)(keys)

ds = Dataset.from_dict(data)
ds = ds.to_iterable_dataset()
ds = ds.with_format("jax")

rng, key = random.split(rng)

model = FunctionEncoder(
    basis_size=8, layer_sizes=(1, 32, 1), activation_function=jax.nn.tanh, key=key
)


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
        point["x"],
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

x = jnp.linspace(-1, 1, 1000).reshape(-1, 1)
y = jnp.polyval(C, x)

example_x = random.uniform(example_key, (10, 1), minval=-1, maxval=1)
example_y = jnp.polyval(C, example_x)

coefficients = model.compute_coefficients(example_x, example_y)
y_pred = model(x, coefficients)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(x, y, label="True")
ax.plot(x, y_pred, label="Predicted")

ax.scatter(example_x, example_y, color="red")

plt.show()
