from typing import Callable

import equinox as eqx
import optax

from function_encoder.function_encoder import FunctionEncoder

import tqdm


def fit(
    model: FunctionEncoder,
    ds,
    loss_function: Callable,
    learning_rate: float = 1e-3,
    gradient_accumulation_steps: int = 50,
):
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=learning_rate),
    )
    # Gradient accumulation
    opt = optax.MultiSteps(opt, every_k_schedule=gradient_accumulation_steps)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def update(model, point, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_function)(model, point)
        updates, opt_state = opt.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    with tqdm.tqdm(enumerate(ds)) as tqdm_bar:
        for i, point in tqdm_bar:
            model, opt_state, loss = update(model, point, opt_state)

            if i % 10 == 0:
                tqdm_bar.set_postfix_str(f"Loss: {loss:.2e}")

    return model
