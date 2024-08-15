import jax.numpy as jnp


def safe_dot(a, b):

    if a.ndim == 1:
        return jnp.dot(a, b)
    else:
        return jnp.dot(a.T, b)
