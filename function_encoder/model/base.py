from jax import tree_util


class BaseModel:

    def __init__(self):
        pass

    def forward(self, x):
        pass

    def _tree_flatten(self):
        children = ()
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


tree_util.register_pytree_node(
    BaseModel, BaseModel._tree_flatten, BaseModel._tree_unflatten
)
