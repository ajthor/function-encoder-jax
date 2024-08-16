from jax import random
import jax.numpy as jnp

from datasets import Dataset

import optax

from function_encoder.model.mlp import MLP
from function_encoder.function_encoder import FunctionEncoder
from function_encoder.coefficients import LeastSquares
from function_encoder.inner_product import EuclideanInnerProduct

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
        x = random.uniform(rng, (10, 1), minval=-1, maxval=1)
        y = jnp.polyval(coefficients, x)

        yield {"x": x, "y": y}


ds = Dataset.from_generator(data_generator)
ds = ds.to_iterable_dataset()
ds = ds.with_format("jax")

inner_product = EuclideanInnerProduct
method = LeastSquares(inner_product=inner_product)

basis_functions = []
k = 8
for _ in range(k):
    rng, params = MLP.init_params(rng)
    basis_functions.append(MLP(params=params))

fe = FunctionEncoder(
    basis_functions=basis_functions,
    method=method,
    inner_product=inner_product,
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(1e-3),
)
optimizer = optax.MultiSteps(optimizer, every_k_schedule=10)  # Gradient accumulation
opt_state = optimizer.init(fe)

for i, point in enumerate(ds):
    fe, opt_state, loss = fe.update(
        point["x"], point["y"], point["x"], point["y"], optimizer, opt_state
    )

    if i % 10 == 0:
        print(f"Loss: {loss}")


# Plot

rng, C = random_polynomial(rng, degree=2)

x = jnp.linspace(-1, 1, 100).reshape(-1, 1)
y = jnp.polyval(C, x)

rng, key = random.split(rng)
example_x = random.uniform(key, (10, 1), minval=-1, maxval=1)
example_y = jnp.polyval(C, example_x)

y_pred = fe.forward(x, example_x, example_y)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(x, y, label="True")
ax.plot(x, y_pred, label="Predicted")

ax.scatter(example_x, example_y, color="red")

plt.show()
