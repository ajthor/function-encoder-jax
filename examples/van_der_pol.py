import jax

jax.config.update("jax_enable_x64", True)

from jax import random
import jax.numpy as jnp

import diffrax
import optax

from function_encoder.model.neural_ode import NeuralODE
from function_encoder.function_encoder import FunctionEncoder, train_function_encoder

import matplotlib.pyplot as plt

# Generate dataset


def van_der_pol(t, x, mu=1):
    return jnp.array([x[1], mu * (1 - x[0] ** 2) * x[1] - x[0]])


def generate_data(key):
    y0_key, mu_key, t_key = random.split(key, 3)

    y0 = random.uniform(y0_key, (2,), minval=-2, maxval=2)
    mu = random.uniform(mu_key, (), minval=0.1, maxval=4)

    ts = jnp.sort(random.uniform(t_key, (1000,), minval=0, maxval=10))
    ts = jnp.concatenate([jnp.array([0.0]), ts])

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(van_der_pol),
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

    return {
        "t0": ts[:-1],
        "tf": ts[1:],
        "x": ys[:-1],
        "y": ys[1:],
        "mu": mu,
    }


ds_size = 1000
rng = random.PRNGKey(0)
rng, key = random.split(rng)
keys = random.split(key, ds_size)
data = jax.vmap(generate_data)(keys)
data = [{k: v[i] for k, v in data.items()} for i in range(ds_size)]


example_data_size = 200


# Create model

rng, key = random.split(rng)

model = FunctionEncoder(
    basis_size=8,
    basis_type=NeuralODE,
    layer_sizes=(2, 32, 2),
    activation_function=jax.nn.tanh,
    key=key,
)


def predict_trajectory(model, y0, ts, coefficients):
    def step_fn(y, t):
        y = model((y, t), coefficients)
        return y, y

    return jax.lax.scan(step_fn, y0, ts)


# Train


def loss_function(model, point):
    ts = jnp.hstack([point["t0"][:, None], point["tf"][:, None]])
    coefficients = model.compute_coefficients(
        (point["x"][:example_data_size], ts[:example_data_size]),
        point["y"][:example_data_size],
    )

    y_pred = predict_trajectory(
        model, point["x"][example_data_size], ts[example_data_size:], coefficients
    )[1]

    return jnp.linalg.norm(y_pred - point["y"][example_data_size:], axis=-1).mean()


model = train_function_encoder(model, data, loss_function)


# Plot

idx = random.randint(rng, (), 0, ds_size)
point = data[idx]

ts = jnp.hstack([point["t0"][:, None], point["tf"][:, None]])
coefficients = model.compute_coefficients(
    (point["x"][:example_data_size], ts[:example_data_size]),
    point["y"][:example_data_size],
)

y_pred = predict_trajectory(
    model, point["x"][example_data_size], ts[example_data_size:], coefficients
)[1]

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(point["y"][:, 0], point["y"][:, 1], label="True")
ax.plot(y_pred[:, 0], y_pred[:, 1], label="Predicted")

# Plot initial data
ax.plot(
    point["y"][:example_data_size, 0], point["y"][:example_data_size, 1], color="red"
)

plt.show()
