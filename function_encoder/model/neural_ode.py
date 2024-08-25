import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import diffrax

from function_encoder.model.mlp import MLP


class Dynamics(eqx.Module):
    mlp: MLP

    def __init__(self, *args, **kwargs):
        self.mlp = MLP(*args, **kwargs)

    def __call__(self, t, x, args):
        return self.mlp(x)


class NeuralODE(eqx.Module):
    dynamics: Dynamics

    def __init__(self, *args, **kwargs):
        self.dynamics = Dynamics(*args, **kwargs)

    def __call__(self, y0_and_time: tuple):
        y0, ts = y0_and_time

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.dynamics),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            adjoint=diffrax.BacksolveAdjoint(),
            saveat=diffrax.SaveAt(t0=True, t1=True),
        )
        return solution.ys[-1]
