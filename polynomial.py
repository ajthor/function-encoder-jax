import jax
from jax import random, tree_util
import jax.numpy as jnp

import equinox as eqx

from datasets import Dataset

import optax

from function_encoder.function_encoder import FunctionEncoder, train

import matplotlib.pyplot as plt

rng = random.PRNGKey(0)


def random_polynomial(rng, degree=2):
    rng, key = random.split(rng)
    coefficients = random.uniform(key, (degree + 1,), minval=-1, maxval=1)

    return rng, coefficients


def data_generator():
    rng = random.PRNGKey(0)
    for i in range(101):

        rng, coefficients = random_polynomial(rng, degree=2)
        x = random.uniform(rng, (23, 1), minval=-1, maxval=1)
        y = jnp.polyval(coefficients, x)

        yield {"x": x, "y": y}


ds = Dataset.from_generator(data_generator)
ds = ds.to_iterable_dataset()
ds = ds.with_format("jax")


model = FunctionEncoder(11, in_size=1, out_size=1, width_size=32, depth=2, key=rng)
model = train(model, ds, steps=100, batch_size=10)


# Plot

rng, C = random_polynomial(rng, degree=2)

x = jnp.linspace(-1, 1, 100).reshape(-1, 1)
y = jnp.polyval(C, x)

rng, key = random.split(rng)
example_x = random.uniform(key, (10, 1), minval=-1, maxval=1)
example_y = jnp.polyval(C, example_x)

y_pred = model.forward(x, example_x, example_y)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(x, y, label="True")
ax.plot(x, y_pred, label="Predicted")

ax.scatter(example_x, example_y, color="red")

plt.show()
