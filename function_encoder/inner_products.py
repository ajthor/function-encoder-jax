import jax.numpy as jnp

import equinox as eqx

from jaxtyping import Array


def euclidean_inner_product(x: Array, y: Array):
    """Compute the inner product."""
    return jnp.dot(x, y)


def L2(x: Array, y: Array):
    """Compute the inner product."""
    return eqx.filter_vmap(euclidean_inner_product, in_axes=(0, 0))(x, y).mean()


def logit_inner_product(x: Array, y: Array):
    """Compute the inner product."""
    _x = x - x.mean(axis=0)
    _y = y - y.mean(axis=0)
    return eqx.filter_vmap(euclidean_inner_product, in_axes=(0, 0))(_x, _y).mean()
