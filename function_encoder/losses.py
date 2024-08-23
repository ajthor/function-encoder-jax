import jax
import jax.numpy as jnp

from jaxtyping import Array


def gram_orthogonality_loss(K: Array):
    """Compute the Gram orthogonality loss."""
    return jnp.linalg.norm(K - jnp.eye(K.shape[0])) ** 2


def gram_normalization_loss(K: Array):
    """Compute the Gram normalization loss."""
    return jnp.linalg.norm(jnp.diagonal(K) - 1) ** 2


def l2_regularizer(coefficients: Array, K: Array):
    """Compute the regularization loss."""
    return jnp.einsum("k,kl,l->", coefficients, K, coefficients)
