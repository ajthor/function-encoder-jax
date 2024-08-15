from abc import ABC, abstractmethod

from jax import vmap, tree_util
import jax.numpy as jnp



class CoefficientMethod(ABC):
    """Abstract base class for coefficient methods."""

    def __init__(self, inner_product):
        self.inner_product = vmap(inner_product, in_axes=(0, None))

    @abstractmethod
    def compute_coefficients(self, G, y):
        """Compute the coefficients."""
        pass

    def _tree_flatten(self):
        children = ()
        aux_data = {"inner_product": self.inner_product}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


class MonteCarloIntegration(CoefficientMethod):
    """Compute the coefficients using Monte Carlo integration."""

    def compute_coefficients(self, G, y):
        """Compute the coefficients using Monte Carlo integration."""

        # Compute the matrix G^T F
        F = jnp.einsum("kmd,md->k", G, y)

        # Compute the coefficients
        coefficients = F / G.shape[0]

        return coefficients

tree_util.register_pytree_node(
    MonteCarloIntegration,
    MonteCarloIntegration._tree_flatten,
    MonteCarloIntegration._tree_unflatten,
)

class LeastSquares(CoefficientMethod):
    """Compute the coefficients using least squares."""

    def compute_coefficients(self, G, y):
        """Compute the coefficients using least squares."""

        # Compute the matrix G^T F
        F = jnp.einsum("kmd,md->k", G, y)

        # Compute the Gram matrix G^T G
        gram = jnp.einsum("kmd,lmd->kl", G, G)

        # Solve the linear system
        coefficients = jnp.linalg.solve(gram, F)

        return coefficients

tree_util.register_pytree_node(
    LeastSquares,
    LeastSquares._tree_flatten,
    LeastSquares._tree_unflatten,
)