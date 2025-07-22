from typing import Tuple, Callable

import jax
import jax.numpy as jnp

import equinox as eqx
from jaxtyping import Array, PRNGKeyArray

from function_encoder.jax.model.mlp import MLP


def rk4_step(func: Callable, x: Array, dt: Array, **ode_kwargs) -> Array:
    """Runge-Kutta 4th order ODE integrator for a single step.

    Args:
        func: ODE function that takes (t, x) and returns dx/dt
        x: Current state vector
        dt: Time step size
        **ode_kwargs: Additional keyword arguments for the ODE function

    Returns:
        The derivative dx computed using RK4
    """
    t = jnp.zeros_like(dt)
    k1 = func(t, x, **ode_kwargs)
    k2 = func(t + dt / 2, x + (dt / 2) * k1, **ode_kwargs)
    k3 = func(t + dt / 2, x + (dt / 2) * k2, **ode_kwargs)
    k4 = func(t + dt, x + dt * k3, **ode_kwargs)
    return (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


class ODEFunc(eqx.Module):
    """A wrapper for a neural network to make it compatible with ODE solvers.

    This class wraps a neural network model to create an ODE function that
    can be used with numerical integrators. It concatenates time and state
    vectors as input to the neural network.

    Args:
        model: Neural network model (typically MLP)
    """

    model: eqx.Module

    def __init__(self, model: eqx.Module):
        self.model = model

    def __call__(self, t, x) -> Array:
        """Compute the time derivative at the current state.

        Args:
            *args: Variable arguments that will be concatenated as input to the neural network

        Returns:
            Time derivative dx/dt
        """
        tx = jnp.concatenate([jnp.atleast_1d(t), x], axis=-1)
        return self.model(tx)


class NeuralODE(eqx.Module):
    """Neural Ordinary Differential Equation model.

    This model uses a neural network to parameterize the dynamics of an ODE
    and integrates it using RK4 numerical integration.

    Args:
        ode_func: ODEFunc wrapping the neural network
        integrator: Integration function (defaults to RK4)
    """

    ode_func: ODEFunc
    integrator: Callable

    def __init__(
        self,
        ode_func: ODEFunc,
        integrator: Callable = rk4_step,
    ):
        self.ode_func = ode_func
        self.integrator = integrator

    def __call__(self, inputs) -> Array:
        """Perform a single integration step.

        Args:
            inputs: Tuple of inputs for the dynamical system

        Returns:
            Change in state after one integration step
        """
        return self.integrator(self.ode_func, *inputs)
