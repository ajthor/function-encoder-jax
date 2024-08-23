import jax

jax.config.update("jax_enable_x64", True)

from jax import random
import jax.numpy as jnp

import optax

from datasets import load_dataset

from function_encoder.model.neural_ode import NeuralODE
from function_encoder.function_encoder import FunctionEncoder

import matplotlib.pyplot as plt

# Load dataset

ds = load_dataset("ajthor/van_der_pol")
ds = ds.with_format("jax")


# Create model

rng = random.PRNGKey(0)
rng, key = random.split(rng)

model = FunctionEncoder(
    basis_size=8,
    basis_type=NeuralODE,
    layer_sizes=(1, 32, 1),
    activation_function=jax.nn.tanh,
    key=key,
)
