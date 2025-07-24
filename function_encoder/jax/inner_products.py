import jax.numpy as jnp

import equinox as eqx

from jaxtyping import Array


@eqx.filter_vmap(in_axes=(0, 0))
def euclidean_inner_product(x: Array, y: Array):
    """Compute the Euclidean inner product between two vectors.

    Args:
        x: Vector of shape (d,)
        y: Vector of shape (d,)

    Returns:
        Scalar inner product <x, y>
    """
    return jnp.dot(x, y)


def standard_inner_product(f: Array, g: Array):
    """Compute the standard inner product between two sets of vectors.

    Takes two sets of vectors, computes the inner product between each pair,
    and returns the average over all pairs.

    Args:
        f: Array of shape (m, d) representing m vectors of dimension d
        g: Array of shape (m, d) representing m vectors of dimension d

    Returns:
        Scalar representing the average inner product
    """
    return euclidean_inner_product(f, g).mean()


def centered_inner_product(f: Array, g: Array):
    """Compute the centered inner product between two sets of vectors.

    Centers both sets of vectors by subtracting their mean, then computes
    the inner product between each pair and returns the average.

    Args:
        f: Array of shape (m, d) representing m vectors of dimension d
        g: Array of shape (m, d) representing m vectors of dimension d

    Returns:
        Scalar representing the average centered inner product
    """
    f_centered = f - f.mean(axis=0)
    g_centered = g - g.mean(axis=0)
    return euclidean_inner_product(f_centered, g_centered).mean()
