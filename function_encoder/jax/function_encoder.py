from typing import Callable, Tuple, Optional

import jax
import jax.numpy as jnp
from jax import random

import equinox as eqx

from jaxtyping import Array, PRNGKeyArray, Float

from function_encoder.jax.model.mlp import MLP
from function_encoder.jax.coefficients import least_squares
from function_encoder.jax.inner_products import standard_inner_product


class BasisFunctions(eqx.Module):
    """A collection of basis functions for function approximation.

    This class creates and manages a set of basis functions (e.g., MLPs, NeuralODEs)
    that can be used to approximate arbitrary functions. Each basis function is
    independently initialized with different random parameters.

    Args:
        basis_size: Number of basis functions to create
        *args: Positional arguments passed to the basis function constructor
        basis_type: Type of basis function to create (e.g., MLP, NeuralODE)
        key: JAX random key for parameter initialization
        **kwargs: Keyword arguments passed to the basis function constructor
    """

    basis_functions: eqx.Module

    def __init__(
        self,
        basis_size: int,
        *args,
        basis_type: type = MLP,
        key: PRNGKeyArray,
        **kwargs,
    ):
        keys = random.split(key, basis_size)
        make_basis_function = lambda key: basis_type(*args, **kwargs, key=key)
        self.basis_functions = eqx.filter_vmap(make_basis_function)(keys)

    def __call__(self, X):
        """Evaluate all basis functions at a single input point.

        Note: This method operates on a single input point. To evaluate multiple
        points, use eqx.filter_vmap to vectorize over the input dimension:

            # Single point evaluation
            g = basis_functions(x)  # Shape: (n_basis,)

            # Multiple point evaluation
            g = eqx.filter_vmap(basis_functions)(X)  # Shape: (n_points, n_basis)

        Args:
            X: Input point (single vector, not batched)

        Returns:
            Array of shape (n_basis,) containing evaluations of all basis functions
        """
        return eqx.filter_vmap(
            lambda model, x: model(x), in_axes=(eqx.if_array(0), None)
        )(self.basis_functions, X)


class FunctionEncoder(eqx.Module):
    """Function encoder that approximates functions using linear combinations of basis functions.

    This class implements a function approximation scheme where unknown functions are
    represented as weighted combinations of basis functions. Given example data points,
    it computes coefficients for the basis functions and can then evaluate the
    approximated function at new points.

    The approximation has the form: f(x) ≈ Σᵢ cᵢ φᵢ(x), where φᵢ are basis functions
    and cᵢ are learned coefficients.

    For residual learning, an optional residual_function can be provided. In this case,
    the approximation becomes: f(x) ≈ r(x) + Σᵢ cᵢ φᵢ(x), where r(x) is the
    residual function. During coefficient computation, gradients are blocked through
    the residual function (equivalent to PyTorch's .detach()).

    Args:
        basis_functions: Collection of basis functions for approximation
        residual_function: Optional residual function (e.g., MLP for average/bias)
        coefficients_method: Method for computing coefficients (e.g., least_squares)
        inner_product: Inner product function for coefficient computation
    """

    basis_functions: BasisFunctions
    residual_function: Optional[eqx.Module]
    coefficients_method: Callable
    inner_product: Callable

    def __init__(
        self,
        basis_functions: BasisFunctions,
        residual_function: Optional[eqx.Module] = None,
        coefficients_method: Callable = least_squares,
        inner_product: Callable = standard_inner_product,
    ) -> None:
        self.basis_functions = basis_functions
        self.residual_function = residual_function
        self.coefficients_method = coefficients_method
        self.inner_product = inner_product

    def compute_coefficients(
        self,
        example_X: Float[Array, "n_examples ..."],
        example_y: Float[Array, "n_examples"],
    ) -> Tuple[Float[Array, "n_basis"], Float[Array, "n_basis n_basis"]]:
        """Compute coefficients for basis functions given example data.

        This method fits the basis functions to the provided example data by computing
        optimal coefficients. If a residual function is present, it subtracts the
        residual (with gradients blocked) before computing coefficients.

        Note: For batch processing multiple functions, use eqx.filter_vmap:
            coeffs, G = eqx.filter_vmap(model.compute_coefficients)(X_batch, y_batch)

        Args:
            example_X: Input points for a single function, shape (n_examples, ...)
            example_y: Output values for a single function, shape (n_examples,)

        Returns:
            coefficients: Learned coefficients, shape (n_basis,)
            G: Gram matrix used in coefficient computation, shape (n_basis, n_basis)
        """
        f = example_y

        if self.residual_function is not None:
            f = f - jax.lax.stop_gradient(
                eqx.filter_vmap(self.residual_function)(example_X)
            )

        g = eqx.filter_vmap(self.basis_functions)(example_X)
        coefficients, G = self.coefficients_method(f, g, self.inner_product)

        return coefficients, G

    def __call__(
        self, X: Float[Array, "..."], coefficients: Float[Array, "n_basis"]
    ) -> Float[Array, ""]:
        """Evaluate the approximated function at a single input point.

        This method computes the function approximation f(x) ≈ Σᵢ cᵢ φᵢ(x) for a
        single input point using precomputed coefficients. If a residual function is
        present, it adds the residual (with gradients blocked) to the result.

        Note: This operates on single points. For multiple point evaluation, use vmap:
            # Single point
            y = model(x, coefficients)

            # Multiple points (same coefficients)
            y = eqx.filter_vmap(model, in_axes=(eqx.if_array(0), None))(X, coefficients)

            # Multiple points with different coefficients per point
            y = eqx.filter_vmap(model)(X, coefficients)

        Args:
            X: Single input point (not batched)
            coefficients: Basis function coefficients, shape (n_basis,)

        Returns:
            Scalar function value at X
        """
        g = self.basis_functions(X)
        y = g.T @ coefficients

        if self.residual_function is not None:
            y = y + jax.lax.stop_gradient(self.residual_function(X))

        return y
