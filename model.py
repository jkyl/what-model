import jax
import jax.numpy as jnp
import flax.linen as nn


def concat_ragged(*args: jax.Array):
    lengths = [a.shape[1] for a in args]
    min_length = min(lengths)
    cropped = []
    for arr, length in zip(args, lengths):
        crop = (length - min_length) // 2
        cropped.append(arr[:, crop:-crop] if crop else arr)
    catted = jnp.concatenate(cropped, axis=2)
    return catted


class ConditionalBatchNorm(nn.Module):

    @nn.compact
    def __call__(self, x: jax.Array, z: jax.Array, train: bool) -> jax.Array:
        x = nn.BatchNorm(use_scale=False, use_bias=False, use_running_average=not train)(x)
        scale = nn.Dense(x.shape[-1])(z).reshape(-1, 1, x.shape[-1])
        bias = nn.Dense(x.shape[-1])(z).reshape(-1, 1, x.shape[-1])
        x = x * scale + bias
        return x


class DilatedBlock(nn.Module):
    ch: int
    depth: int
    kernel_size: int

    def shortcut(self, x: jax.Array, dilation: int):
        p = (self.kernel_size // 2) * dilation
        x = x[:, p:-p]
        if x.shape[-1] != self.ch:
            x = nn.Conv(features=self.ch, kernel_size=(1,), use_bias=False)(x)
        return x

    @nn.compact
    def __call__(self, x: jax.Array, z: jax.Array, train: bool) -> jax.Array:
        for i in range(self.depth + 1):
            dilation = 2 ** (i % self.depth)
            x0 = self.shortcut(x, dilation)
            x = ConditionalBatchNorm()(x, z, train)
            x = nn.relu(x)
            x = nn.Conv(
                features=self.ch,
                kernel_size=(self.kernel_size,),
                kernel_dilation=dilation,
                padding="VALID",
                use_bias=False,
            )(x)
            x += x0
        return x


class DilatedDenseNet(nn.Module):
    ch: int
    depth: int
    kernel_size: int
    num_blocks: int
    embedding_dim: int

    def __post_init__(self) -> None:
        self.pad = self.num_blocks * (self.kernel_size - 1) * 2 ** self.depth
        self.p = self.pad // 2
        dummy_xt = jnp.ones((1, self.pad + 1, 1))
        dummy_t = jnp.ones((1,), dtype=jnp.int32)
        self.dummy_args = (dummy_xt, dummy_t)
        super().__post_init__()

    def _mlp(self, x):
        x = x.reshape(-1, 1)
        x = nn.Dense(self.embedding_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.embedding_dim)(x)
        x = nn.relu(x)
        return x

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        t: jax.Array,
        train: bool,
    ) -> jax.Array:
        ch_in = x.shape[-1]
        z = self._mlp(t)
        cat = [x]
        for _ in range(self.num_blocks):
            cat.append(DilatedBlock(self.ch, self.depth, self.kernel_size)(cat[-1], z, train))
        x = concat_ragged(*cat)
        x = ConditionalBatchNorm()(x, z, train)
        x = nn.relu(x)
        x = nn.Conv(features=ch_in, kernel_size=(1,), use_bias=False)(x)
        return x
