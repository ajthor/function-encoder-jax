from typing import Callable, Mapping

from jax import random
import jax.numpy as jnp

import equinox as eqx

from jaxtyping import Array, PRNGKeyArray

from function_encoder.model.mlp import MLP
from function_encoder.function_encoder import FunctionEncoder


# class OperatorEncoder(eqx.Module):
#     source_model: FunctionEncoder
#     target_model: FunctionEncoder

#     operator_model: eqx.Module

#     def __init__(
#         self,
#         source_model: FunctionEncoder,
#         target_model: FunctionEncoder,
#         operator_model: eqx.Module,
#         key,
#     ):
#         self.source_model = source_model
#         self.target_model = target_model
#         self.operator_model = operator_model

#     def compute_coefficients(self, example_X: Array, example_y: Array):
#         coefficients_source = self.source_model.compute_coefficients(
#             example_X, example_y
#         )
#         coefficients_target = self.operator_model(coefficients_source)
#         return coefficients_target

#     def __call__(self, X: Array, coefficients: Array):
#         return self.target_model(X, coefficients)


class OperatorEncoder(eqx.Module):
    source_model: FunctionEncoder
    target_model: FunctionEncoder

    operator: MLP

    def __init__(
        self,
        source_config: Mapping,
        target_config: Mapping,
        operator_config: Mapping,
        *args,
        key: PRNGKeyArray,
        **kwargs
    ):
        source_key, target_key, operator_key = random.split(key, 3)
        self.source_model = FunctionEncoder(**source_config, key=source_key)
        self.target_model = FunctionEncoder(**target_config, key=target_key)
        self.operator = MLP(**operator_config, key=operator_key)

    def compute_coefficients(self, example_X: Array, example_y: Array):
        coefficients_source = self.source_model.compute_coefficients(
            example_X, example_y
        )
        coefficients_target = self.operator(coefficients_source)
        return coefficients_target

    def __call__(self, X: Array, coefficients: Array):
        return self.target_model(X, coefficients)


class LinearOperatorEncoder(eqx.Module):
    source_model: FunctionEncoder
    target_model: FunctionEncoder

    operator: Array

    def __init__(
        self,
        source_config: Mapping,
        target_config: Mapping,
        *args,
        key: PRNGKeyArray,
        **kwargs
    ):

        source_key, target_key, operator_key = random.split(key, 3)
        self.source_model = FunctionEncoder(**source_config, key=source_key)
        self.target_model = FunctionEncoder(**target_config, key=target_key)
        self.operator = random.normal(
            operator_key, (source_config["basis_size"], target_config["basis_size"])
        )

    def compute_coefficients(self, example_X: Array, example_y: Array):
        coefficients_source = self.source_model.compute_coefficients(
            example_X, example_y
        )
        coefficients_target = jnp.dot(coefficients_source, self.operator)
        return coefficients_target

    def __call__(self, X: Array, coefficients: Array):
        return self.target_model(X, coefficients)
