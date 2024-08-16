from abc import ABC, abstractmethod
from typing import Any

from jax import tree_util
import jax.numpy as jnp


from function_encoder.utils.safe_dot import safe_dot


def EuclideanInnerProduct(X, Y):
    return jnp.dot(X, Y)
