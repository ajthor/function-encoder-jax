from typing import Tuple, Callable, Optional, Dict, Any

import jax
import jax.numpy as jnp

import equinox as eqx
from jaxtyping import Array, Float, Scalar

from function_encoder.jax.model.mlp import MLP


def rk4_step(
    func: Callable[[Scalar, Float[Array, "..."]], Array],
    x: Float[Array, "..."],
    dt: Scalar,
    **ode_kwargs
) -> Float[Array, "..."]:
    """Runge-Kutta 4th order ODE integrator for a single step.

    Args:
        func (Callable): ODE function that takes (t, x) and returns dx/dt
        x (Array): Current state vector
        dt (Scalar): Time step size
        **ode_kwargs: Additional keyword arguments for the ODE function

    Returns:
        Array: The derivative dx computed using RK4
    """
    t = jnp.zeros_like(dt)
    k1 = func(t, x, **ode_kwargs)
    k2 = func(dt / 2, x + (dt / 2) * k1, **ode_kwargs)
    k3 = func(dt / 2, x + (dt / 2) * k2, **ode_kwargs)
    k4 = func(dt, x + dt * k3, **ode_kwargs)
    return (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


class ODEFunc(eqx.Module):
    """A wrapper for a neural network to make it compatible with ODE solvers.

    This class wraps a neural network model to create an ODE function that
    can be used with numerical integrators. It concatenates time and state
    vectors as input to the neural network.

    Args:
        model (eqx.Module): The neural network model (typically MLP).
    """

    model: eqx.Module

    def __init__(self, model: eqx.Module):
        self.model = model

    def __call__(self, t: Scalar, x: Float[Array, "..."]) -> Float[Array, "..."]:
        """Compute the time derivative at the current state.

        Args:
            t (Scalar): Current time
            x (Array): Current state vector

        Returns:
            Array: The time derivative dx/dt at the current state
        """
        tx = jnp.concatenate([jnp.atleast_1d(t), x], axis=-1)
        return self.model(tx)


class NeuralODE(eqx.Module):
    """Neural Ordinary Differential Equation model.

    This model uses a neural network to parameterize the dynamics of an ODE
    and integrates it using numerical integration methods.

    Args:
        ode_func (ODEFunc): The vector field function wrapping a neural network
        integrator (Callable): The ODE solver (e.g., `rk4_step`). Defaults to rk4_step.
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

    def __call__(
        self,
        inputs: Tuple[Float[Array, "..."], Scalar],
        ode_kwargs: Optional[Dict[str, Any]] = {},
    ) -> Float[Array, "..."]:
        """Solve the initial value problem for a single time step.

        Args:
            inputs (tuple): A tuple containing (y0, dt), where:
                y0 (Array): Initial condition/current state
                dt (Scalar): Time step size
            ode_kwargs (dict, optional): Additional integrator arguments. Defaults to {}.

        Returns:
            Array: Solution of the ODE at the next time step (change in state).
        """
        return self.integrator(self.ode_func, *inputs, **ode_kwargs)
