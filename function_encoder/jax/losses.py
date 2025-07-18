import jax.numpy as jnp

from jaxtyping import Array


def basis_orthogonality_loss(K: Array):
    """Compute the basis orthogonality loss.
    
    Penalizes off-diagonal elements of the Gram matrix to encourage
    orthogonal basis functions. Does not enforce unit norm.
    
    Args:
        K: Gram matrix of shape (k, k) where k is number of basis functions
        
    Returns:
        Scalar loss value
    """
    # Zero out diagonal, penalize off-diagonal elements
    off_diagonal = K - jnp.diag(jnp.diag(K))
    return jnp.linalg.norm(off_diagonal, ord="fro") ** 2


def basis_normalization_loss(K: Array):
    """Compute the basis normalization loss.
    
    Penalizes diagonal elements of the Gram matrix being far from one,
    encouraging unit norm basis functions.
    
    Args:
        K: Gram matrix of shape (k, k) where k is number of basis functions
        
    Returns:
        Scalar loss value
    """
    return ((jnp.diag(K) - 1) ** 2).mean()


def basis_orthonormality_loss(K: Array):
    """Compute the basis orthonormality loss.
    
    Penalizes the Gram matrix being far from the identity matrix,
    encouraging orthonormal basis functions (both orthogonal and unit norm).
    
    Args:
        K: Gram matrix of shape (k, k) where k is number of basis functions
        
    Returns:
        Scalar loss value
    """
    eye = jnp.eye(K.shape[0])
    return (jnp.linalg.norm(K - eye, ord="fro") ** 2).mean()


def residual_loss(model, inputs, targets):
    """Compute the residual loss.
    
    Computes mean squared error between the model's residual prediction
    and the target values.
    
    Args:
        model: Model with residual_function method
        inputs: Input data
        targets: Target values
        
    Returns:
        Mean squared error loss
    """
    predictions = model.residual_function(inputs)
    return jnp.mean((predictions - targets) ** 2)
