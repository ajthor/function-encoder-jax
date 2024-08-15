from jax import jit, random, tree_util
import jax.numpy as jnp

from function_encoder.model.base import BaseModel


class MLP(BaseModel):
    """Multi-layer perceptron."""

    def __init__(self, params=None, activation=jnp.tanh):
        super().__init__()

        self.activation = activation

        if params is not None:
            self.params = params
        else:
            _, self.params = MLP.init_params(random.PRNGKey(0))

    @staticmethod
    def init_params(rng, layer_sizes=[1, 32, 1]):
        """Initialize the parameters of the MLP."""

        params = []
        # Up until the last layer, we have weights and biases
        for n_in, n_out in zip(layer_sizes[:-2], layer_sizes[1:-1]):
            rng, w_key, b_key = random.split(rng, 3)
            C = jnp.sqrt(layer_sizes[0])

            w = random.uniform(w_key, (n_in, n_out), minval=-C, maxval=C)
            b = random.uniform(b_key, (n_out), minval=-C, maxval=C)

            params.append((w, b))

        # Last layer only has weights
        rng, w_key = random.split(rng)
        w = random.uniform(
            w_key, (layer_sizes[-2], layer_sizes[-1]), minval=-C, maxval=C
        )

        params.append((w,))

        return rng, params

    @jit
    def forward(self, X):
        """Forward pass."""

        for w, b in self.params[:-1]:
            y = jnp.dot(X, w) + b
            X = self.activation(y)

        (w,) = self.params[-1]

        y = jnp.dot(X, w)

        return y

    def _tree_flatten(self):
        children = (self.params,)
        aux_data = {
            "activation": self.activation,
        }
        return (children, aux_data)

tree_util.register_pytree_node(
    MLP, MLP._tree_flatten, MLP._tree_unflatten
)