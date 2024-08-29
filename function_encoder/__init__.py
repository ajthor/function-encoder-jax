from .function_encoder import (
    monte_carlo_integration as monte_carlo_integration,
    least_squares as least_squares,
    BasisFunctions as BasisFunctions,
    FunctionEncoder as FunctionEncoder,
    ResidualFunctionEncoder as ResidualFunctionEncoder,
)

from .operator_encoder import (
    EigenOperatorEncoder as EigenOperatorEncoder,
    SVDOperatorEncoder as SVDOperatorEncoder,
)

from .utils.training import fit as fit

from .losses import (
    basis_normalization_loss as basis_normalization_loss,
    basis_orthogonality_loss as basis_orthogonality_loss,
    l2_regularizer as l2_regularizer,
)

from .model.mlp import MLP as MLP
from .model.neural_ode import NeuralODE as NeuralODE
