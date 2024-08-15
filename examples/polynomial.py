from jax import random
import jax.numpy as jnp

from datasets import Dataset

import optax

from function_encoder.model.mlp import MLP
from function_encoder.function_encoder import FunctionEncoder
from function_encoder.coefficients import LeastSquares
from function_encoder.inner_product import EuclideanInnerProduct


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


ds = Dataset.from_generator(data_generator).with_format("jax")


inner_product = EuclideanInnerProduct
method = LeastSquares(inner_product=inner_product)

basis_functions = [MLP() for _ in range(7)]

fe = FunctionEncoder(
    basis_functions=basis_functions,
    method=method,
    inner_product=inner_product,
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(1e-3),
)
opt_state = optimizer.init(fe)