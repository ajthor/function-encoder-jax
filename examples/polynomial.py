import jax

jax.config.update("jax_enable_x64", True)

from jax import random
import jax.numpy as jnp

import equinox as eqx
import optax

from datasets.polynomial import PolynomialDataset

from function_encoder.jax.losses import basis_normalization_loss
from function_encoder.jax.function_encoder import FunctionEncoder, BasisFunctions
from function_encoder.jax.utils.training import train_step

import tqdm

import matplotlib.pyplot as plt

# Load dataset

dataset = PolynomialDataset(n_points=100, n_example_points=10)
# batch_function = lambda key: eqx.filter_vmap(dataset)(random.split(key, 1))

# Create model

rng = random.PRNGKey(0)
rng, key = random.split(rng)

basis_functions = BasisFunctions(basis_size=8, layer_sizes=(1, 32, 1), key=key)
model = FunctionEncoder(basis_functions=basis_functions)

# Train


def loss_function(model, point):
    X, y, example_X, example_y = point
    coefficients, G = model.compute_coefficients(example_X, example_y)
    y_pred = eqx.filter_vmap(model, in_axes=(eqx.if_array(0), None))(X, coefficients)
    pred_loss = optax.squared_error(y, y_pred).mean()
    norm_loss = basis_normalization_loss(G)
    return pred_loss + norm_loss


# model = fit(model, ds["train"], loss_function)
every_k_schedule = 50
opt = optax.MultiSteps(
    optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(1e-3),
    ),
    every_k_schedule=every_k_schedule,
)
opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

num_epochs = 1000
update = eqx.filter_jit(train_step)
with tqdm.tqdm(range(num_epochs * every_k_schedule)) as tqdm_bar:
    for epoch in tqdm_bar:
        rng, key = random.split(rng)
        point = dataset(key)
        model, opt_state, loss = update(model, opt, opt_state, point, loss_function)

        if epoch % 10 == 0:
            tqdm_bar.set_postfix_str(f"Loss: {loss:.2e}")


# Plot

rng, key = random.split(rng)
point = dataset(key)

X, y, example_X, example_y = point

idx = jnp.argsort(X, axis=0).flatten()
X = X[idx]
y = y[idx]

coefficients, _ = model.compute_coefficients(example_X, example_y)
y_pred = eqx.filter_vmap(model, in_axes=(eqx.if_array(0), None))(X, coefficients)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(X, y, label="True")
ax.scatter(X, y, label="Data", color="red")

ax.plot(X, y_pred, label="Predicted")

plt.show()
