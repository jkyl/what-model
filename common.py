import jax
import jax.numpy as jnp

from functools import partial
from typing import Tuple


#@partial(jax.jit, static_argnums=2)
def batch_crops(data: jax.Array, starts: jax.Array, length: int) -> Tuple[jax.Array, jax.Array]:

    # Translate the crop indices to (potentially OOB) positions.
    indices = starts[:, None] + jnp.arange(length)

    # This value is guaranteed to be OOB for this array.
    oob = data.shape[0]
    
    # Create a mask of the indices that are OOB.
    oob_mask = jnp.logical_or(indices < 0, indices >= oob)

    # Replace all OOB indices with single OOB value.
    indices = jnp.where(oob_mask, oob, indices)

    print(oob, indices)

    # Use jnp.take instead of jax.lax.dynamic_slice to handle zero-padding.
    batch = jnp.take(data, indices, mode="fill", fill_value=0)

    print(batch[0])
    
    return batch, oob_mask
