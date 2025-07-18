from typing import Callable, Tuple, Iterable, Any

import equinox as eqx
import optax
from jaxtyping import Float, Scalar


def train_step(
    model: eqx.Module,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    point: Any,
    loss_function: Callable[[eqx.Module, Any], Scalar],
) -> Tuple[eqx.Module, optax.OptState, Float]:
    """Perform a single training step.

    Args:
        model: The model to train
        optimizer: Optax optimizer
        opt_state: Optimizer state
        point: Training point (single function's input/output pairs)
        loss_function: Function that computes loss given (model, point)

    Returns:
        Tuple of (updated_model, updated_opt_state, loss_value)
    """
    loss, grads = eqx.filter_value_and_grad(loss_function)(model, point)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def test_eval(
    model: eqx.Module,
    dataset: Iterable[Any],
    loss_function: Callable[[eqx.Module, Any], Scalar],
) -> Float:
    """Evaluate model on entire dataset and return average loss.

    Args:
        model: The model to evaluate
        dataset: Dataset to evaluate on (iterable of function input/output pairs)
        loss_function: Function that computes loss given (model, point)

    Returns:
        Average loss across all points in the dataset
    """
    total_loss = 0.0
    count = 0
    
    for point in dataset:
        loss = loss_function(model, point)
        total_loss += loss
        count += 1
    
    return total_loss / count if count > 0 else 0.0
