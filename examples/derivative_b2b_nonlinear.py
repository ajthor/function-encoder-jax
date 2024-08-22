import jax

jax.config.update("jax_enable_x64", True)

from jax import random
import jax.numpy as jnp

from datasets import load_dataset, Dataset

import equinox as eqx
import optax

from function_encoder.model.mlp import MLP
from function_encoder.function_encoder import FunctionEncoder, train_function_encoder

import matplotlib.pyplot as plt

import tqdm


ds = load_dataset("ajthor/derivative")
ds = ds.with_format("jax")

rng = random.PRNGKey(0)
source_key, target_key, operator_key = random.split(rng, 3)


source_encoder = FunctionEncoder(
    basis_size=8,
    layer_sizes=(1, 32, 1),
    activation_function=jax.nn.tanh,
    key=source_key,
)

target_encoder = FunctionEncoder(
    basis_size=8,
    layer_sizes=(1, 32, 1),
    activation_function=jax.nn.tanh,
    key=target_key,
)

operator = MLP(
    layer_sizes=(8, 32, 32, 8),
    activation_function=jax.nn.tanh,
    key=operator_key,
)


# Train


# Train the source encoder.
def source_loss_function(model, point):
    coefficients = model.compute_coefficients(point["X"], point["f"])
    f_pred = model(point["X"], coefficients)
    return optax.l2_loss(point["f"], f_pred).mean()


# source_encoder = train_function_encoder(
#     source_encoder, ds["train"].take(1000), source_loss_function
# )


# Train the target encoder.
def target_loss_function(model, point):
    coefficients = model.compute_coefficients(point["Y"], point["Tf"])
    Tf_pred = model(point["Y"], coefficients)
    return optax.l2_loss(point["Tf"], Tf_pred).mean()


# target_encoder = train_function_encoder(
#     target_encoder, ds["train"].take(1000), target_loss_function
# )


# Train the operator.


def operator_loss_function(model, point):
    source_coefficients = source_encoder.compute_coefficients(point["X"], point["f"])
    target_coefficients_pred = model(source_coefficients)
    target_coefficients = target_encoder.compute_coefficients(point["Y"], point["Tf"])
    return optax.l2_loss(target_coefficients, target_coefficients_pred).mean()


opt = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
opt_state = opt.init(eqx.filter(operator, eqx.is_inexact_array))


@eqx.filter_jit
def update(model, point, opt_state):
    loss, grads = eqx.filter_value_and_grad(operator_loss_function)(model, point)
    updates, opt_state = opt.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


with tqdm.tqdm(enumerate(ds["train"]), total=ds["train"].num_rows) as tqdm_bar:
    for i, point in tqdm_bar:
        operator, opt_state, loss = update(operator, point, opt_state)

        if i % 10 == 0:
            tqdm_bar.set_postfix_str(f"Loss: {loss:.2e}")


# Plot

point = ds["train"].take(1)[0]

source_coefficients = source_encoder.compute_coefficients(point["X"], point["f"])
target_coefficients = operator(source_coefficients)
Tf_pred = target_encoder(point["Y"], target_coefficients)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(point["Y"], point["Tf"], label="True")
ax.plot(point["Y"], Tf_pred, label="Predicted")

plt.show()
