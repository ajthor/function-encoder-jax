import jax
from jax import jit, vmap, random, tree_util
import jax.numpy as jnp

from datasets import Dataset

import optax

from datasets.polynomial import (
    random_polynomial,
    random_polynomial_dataset,
    data_generator,
)
from function_encoder.function_encoder import FunctionEncoder, train

import matplotlib.pyplot as plt

rng = random.PRNGKey(0)

rng, key = random.split(rng)


ds = Dataset.from_generator(data_generator)
ds = ds.to_iterable_dataset()
ds = ds.with_format("jax")

model = FunctionEncoder(
    basis_size=100,
    layer_sizes=(1, 32, 1),
    activation_function=jax.nn.tanh,
    key=key,
)
model = train(model, ds, batch_size=10)


# Plot

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

for _ax in ax:

    rng, poly_key, data_key = random.split(rng, 3)
    C = random_polynomial(poly_key)
    data = random_polynomial_dataset(data_key, C, n_samples=1000, n_examples=10)
    X, y, example_X, example_y = (
        data["X"],
        data["y"],
        data["example_X"],
        data["example_y"],
    )

    coefficients = model.compute_coefficients(example_X, example_y)
    y_pred = model(X, coefficients)

    _ax.plot(X, y, label="True")
    _ax.plot(X, y_pred, label="Predicted")

    _ax.scatter(example_X, example_y, color="red")

plt.show()
