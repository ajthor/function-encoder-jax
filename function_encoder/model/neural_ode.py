from functools import partial

from jax import jit, vmap, random, tree_util
import jax.numpy as jnp
from jaxtyping import Array, Key

import equinox as eqx

import diffrax

from function_encoder.model.mlp import MLP


# class Dynamics(eqx.Module):
#     mlp: MLP

#     def __init__(self, *args, **kwargs):
#         self.mlp = MLP(*args, **kwargs)

#     def __call__(self, t, x):
#         return self.mlp(x)


class NeuralODE(eqx.Module):
    dynamics: MLP

    def __init__(self, *args, **kwargs):
        self.dynamics = MLP(*args, **kwargs)

    def __call__(self, ts, y0):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(lambda t, y: self.dynamics(y)),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys
