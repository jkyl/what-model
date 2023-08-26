import jax
import jax.numpy as jnp

from functools import partial


@partial(jax.jit, static_argnums=2)
def batch_crops(data: jax.Array, starts: jax.Array, length: int):

    # Crops will use jnp.take with shifted versions of this index array.
    indices = jnp.arange(length)

    # Value is guaranteed to be OOB for this array.
    oob = data.shape[0]

    def zero_padded_crop(start):

        # Translate the crop indices to (potentially OOB) positions.
        shifted_indices = start + indices

        # Replace negative indices with OOB index.
        shifted_indices = jnp.where(shifted_indices < 0, oob, shifted_indices)

        # Use jnp.take instead of jax.lax.dynamic_slice to handle OOB zero-padding.
        return jnp.take(data, shifted_indices, mode="fill", fill_value=0)

    # vmap the function over all starts.
    return jax.vmap(zero_padded_crop)(starts)
