from functools import partial

from jax import jit, vmap, random, tree_util
import jax.numpy as jnp

import equinox as eqx
