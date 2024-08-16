import jax
from jax import random, tree_util
import jax.numpy as jnp

import equinox as eqx

from datasets import Dataset

import optax

# from function_encoder.model.mlp import MLP
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
        x = random.uniform(rng, (23, 1), minval=-1, maxval=1)
        y = jnp.polyval(coefficients, x)

        yield {"x": x, "y": y}


ds = Dataset.from_generator(data_generator)
ds = ds.to_iterable_dataset()
ds = ds.with_format("jax")

inner_product = EuclideanInnerProduct
method = LeastSquares(inner_product=inner_product)

# rng, params = MLP.init_params(rng, n_basis=11, layer_sizes=[1, 32, 1])
# basis_functions = MLP(params=params)


@eqx.filter_vmap
def make_ensemble(key):
    return eqx.nn.MLP(2, 2, 2, 2, key=key)


key = random.PRNGKey(0)
keys = random.split(key, 11)
mlp_ensemble = make_ensemble(keys)


class BasisFunction(eqx.Module):
    basis_functions: eqx.Module

    def __init__(self, n_basis, key):
        keys = random.split(key, n_basis)

        def init(key):
            return eqx.nn.MLP(1, 1, 32, 2, key=key)

        self.basis_functions = eqx.filter_vmap(init)(keys)

    def __call__(self, X, example_X, example_y):
        # coefficients =
        G = eqx.filter_vmap(self.basis_functions, in_axes=(0, None))(X)
        return pred


basis_functions = BasisFunction(11, key)

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
