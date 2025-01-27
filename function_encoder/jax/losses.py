import jax
import jax.numpy as jnp

from jaxtyping import Array

import equinox as eqx

from function_encoder.jax.function_encoder import BasisFunctions


def basis_orthogonality_loss(basis_functions: BasisFunctions, X: Array):
    """Compute the basis orthogonality loss."""
    G = eqx.filter_vmap(basis_functions)(X)
    K = jnp.einsum("mkd,mld->kl", G, G)
    return jnp.linalg.norm(K - jnp.eye(K.shape[0]), ord="fro") ** 2


def basis_normalization_loss(basis_functions: BasisFunctions, X: Array):
    """Compute the basis normalization loss."""
    G = eqx.filter_vmap(basis_functions)(X)
    K = jnp.einsum("mkd,mld->kl", G, G)
    return jnp.linalg.norm(jnp.diagonal(K) - 1) ** 2


def l2_regularizer(coefficients: Array, K: Array):
    """Compute the regularization loss."""
    return jnp.einsum("k,kl,l->", coefficients, K, coefficients)
