from typing import Tuple

import jax

from jax import random
import jax.numpy as jnp

from jax.experimental.ode import odeint

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import diffrax
import optax

from jaxtyping import Array

from function_encoder.model.mlp import MLP

import tqdm


class Dynamics(eqx.Module):
    mlp: MLP

    def __init__(self, *args, key: random.PRNGKey, **kwargs):
        self.mlp = MLP(*args, key=key, **kwargs)

    def __call__(self, t, x, args):
        return self.mlp(jnp.hstack([t, x]))


class NeuralODE(eqx.Module):
    dynamics: Dynamics

    def __init__(self, *args, key: random.PRNGKey, **kwargs):
        self.dynamics = Dynamics(*args, key=key, **kwargs)

    def __call__(self, y0_and_time: Tuple[Array, Array]):
        y0, ts = y0_and_time

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.dynamics),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            # stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            adjoint=diffrax.BacksolveAdjoint(),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys[-1]
