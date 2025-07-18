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

    def __call__(self, *args) -> Array:
        """Compute the time derivative at the current state.
        
        Args:
            *args: Variable arguments that will be concatenated as input to the neural network
            
        Returns:
            Time derivative dx/dt
        """
        # Concatenate all inputs for neural network input
        inputs = jnp.concatenate([jnp.atleast_1d(arg) for arg in args])
        return self.model(inputs)


class NeuralODE(eqx.Module):
    """Neural Ordinary Differential Equation model.
    
    This model uses a neural network to parameterize the dynamics of an ODE
    and integrates it using RK4 numerical integration.
    
    Args:
        layer_sizes: Tuple of layer sizes for the neural network
        activation_function: Activation function for the neural network
        integrator: Integration function (defaults to RK4)
        key: JAX random key for initialization
        **kwargs: Additional arguments passed to the neural network
    """
    ode_func: ODEFunc
    integrator: Callable

    def __init__(
        self,
        layer_sizes: Tuple[int, ...],
        *,
        activation_function: Callable = jax.nn.relu,
        integrator: Callable = rk4_step,
        key: PRNGKeyArray,
        **kwargs,
    ):
        # Create the neural network model
        model = MLP(layer_sizes, activation_function=activation_function, key=key, **kwargs)
        
        # Wrap it in ODEFunc
        self.ode_func = ODEFunc(model)
        self.integrator = integrator

    def __call__(self, inputs: Tuple) -> Array:
        """Perform a single integration step.
        
        Args:
            inputs: Tuple of inputs for the dynamical system
            
        Returns:
            State after one integration step
        """
        # Use the integrator to compute the derivative and update state
        dx = self.integrator(self.ode_func, *inputs)
        # Assume the first input is the current state
        return inputs[0] + dx
