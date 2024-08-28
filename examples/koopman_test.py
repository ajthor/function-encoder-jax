"""Koopman operator test.

Given a dynamical system,

    x_{t+1} = f(x_t),

the Koopman operator K_f of the system is a composition operator,

    K_f g = g o f.

In other words, K_f g(x_t) = g(f(x_t)) = g(x_{t+1}). 

We consider learning the Koopman operator using trajectory data (x_t, x_{t+1}) and 
a random set of observables g. 

"""

from functools import partial

import jax

jax.config.update("jax_enable_x64", True)

from jax import random
import jax.numpy as jnp

from datasets import load_dataset, Dataset

import equinox as eqx
import diffrax
import optax

from function_encoder.losses import basis_normalization_loss
from function_encoder.function_encoder import FunctionEncoder, train_model

import matplotlib.pyplot as plt

import tqdm


# Load dataset


def dynamics(t, x, mu=1):
    return jnp.array([x[1], mu * (1 - x[0] ** 2) * x[1] - x[0]])


def generate_data(key):
    # y0_key, mu_key, t_key = random.split(key, 3)

    y0 = random.uniform(key, (2,), minval=-2, maxval=2)
    # mu = random.uniform(mu_key, (), minval=0.1, maxval=4)
    mu = 2

    # ts = jnp.sort(random.uniform(t_key, (1000 - 1,), minval=0, maxval=10))
    # ts = jnp.concatenate([jnp.array([0.0]), ts])
    ts = jnp.linspace(0, 5, 500)

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(dynamics),
        diffrax.Tsit5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=y0,
        args=mu,
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
        saveat=diffrax.SaveAt(ts=ts),
    )
    ys = solution.ys

    return {"t": ts, "x": ys, "mu": mu}


ds_size = 1000
rng = random.PRNGKey(0)
rng, key = random.split(rng)
keys = random.split(key, ds_size)
data = jax.vmap(generate_data)(keys)

ds = Dataset.from_dict(data).train_test_split(test_size=0.1, shuffle=False)
ds = ds.with_format("jax")


def shorten_traj(point, length):
    point["x"] = point["x"][:length]
    point["t"] = point["t"][:length]
    return point


# Create model

rng, model_key = random.split(rng)

basis_size = 8
model = FunctionEncoder(
    basis_size=basis_size,
    layer_sizes=(2, 64, 2),
    activation_function=jax.nn.tanh,
    key=model_key,
)

batched_model = eqx.filter_vmap(model, in_axes=(eqx.if_array(0), None))


# Train


# Train the function encoder to represent the identity function.
def function_encoder_loss_function(model, point):
    coefficients = model.compute_coefficients(point["x"], point["x"])
    y_pred = batched_model(point["x"], coefficients)
    pred_loss = optax.squared_error(y_pred, point["x"]).mean()
    norm_loss = basis_normalization_loss(model.basis_functions, point["x"])
    return norm_loss


model = train_model(model, ds["train"], function_encoder_loss_function)


data = ds["train"].take(10)
x = jnp.astype(data["x"], jnp.float64)
t = jnp.astype(data["t"], jnp.float64)

# Join the data.
x = data["x"][:, 1:].reshape(-1, 2)
y = data["x"][:, :-1].reshape(-1, 2)


identity_coefficients = model.compute_coefficients(x, x)

G = eqx.filter_vmap(model.basis_functions)(x)
M = jnp.einsum("mkd,mld->kl", G, G)
L = jnp.linalg.cholesky(M)
rng, key = random.split(rng)
coefficients = jnp.dot(L, random.normal(key, (basis_size, 20)))  # Random coefficients

CTG = eqx.filter_vmap(batched_model, in_axes=(None, eqx.if_array(1)))

rng, key = random.split(rng)
K = random.uniform(key, (basis_size, basis_size))  # Initialize the Koopman operator

# Now do SGD to minimize || (Kc)^T G(x) - c^T G(y) ||^2
solver = optax.adam(learning_rate=1e-2)
opt_state = solver.init(K)


def loss_function(K, x, y, coefficients):
    K_pred = CTG(x, jnp.dot(K, coefficients)).reshape(2, -1)
    pred_loss = optax.squared_error(K_pred, CTG(y, coefficients).reshape(2, -1)).mean()
    return pred_loss


@eqx.filter_jit
def update(K, x, y, coefficients, opt_state):
    loss, grad = jax.value_and_grad(loss_function)(K, x, y, coefficients)
    updates, opt_state = solver.update(grad, opt_state)
    K = optax.apply_updates(K, updates)
    return K, loss


with tqdm.tqdm(range(1000)) as tqdm_bar:
    for i in tqdm_bar:
        K, loss = update(K, x, y, coefficients, opt_state)

        if i % 10 == 0:
            tqdm_bar.set_postfix_str(f"Loss: {loss:.2e}")


# Plot

data = ds["train"][42]
x = jnp.astype(data["x"], jnp.float64)
t = jnp.astype(data["t"], jnp.float64)
# x = data["x"]
# t = data["t"]

# rng, key = random.split(rng)
# coefficients = jnp.dot(L, random.normal(key, (8,)))
coefficients = coefficients[:, 0]
# coefficients = identity_coefficients
y = batched_model(x, coefficients)
y_pred = batched_model(x, jnp.dot(K, coefficients))

# fig = plt.figure()
# ax = fig.add_subplot(111)

# ax.plot(t, y, label="Original")
# ax.plot(t, y_pred, label="Predicted")

# plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(y[:, 0], y[:, 1], label="Original")
ax.plot(y_pred[:, 0], y_pred[:, 1], label="Predicted")

plt.show()
