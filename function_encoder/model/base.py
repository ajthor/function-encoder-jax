from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Abstract base class for models."""

    @abstractmethod
    def __call__(self, x):
        """Evaluate the model."""
        pass

    def tree_flatten(self):
        children = ()
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
