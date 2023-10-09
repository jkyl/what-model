import jax
import jax.numpy as jnp
import flax.linen as nn

from typing import Tuple


class Affine(nn.Module):

    @nn.compact
    def __call__(self, x: jax.Array, z: jax.Array) -> jax.Array:
        scale = nn.Dense(x.shape[-1], bias_init=nn.initializers.ones_init())(z)
        bias = nn.Dense(x.shape[-1], bias_init=nn.initializers.zeros_init())(z)
        x = x * scale[:, None] + bias[:, None]
        return x


class DilatedBlock(nn.Module):
    ch: int
    depth: int
    kernel_size: int

    def shortcut(self, x: jax.Array, dilation: int):
        p = (self.kernel_size // 2) * dilation
        x = x[:, p:-p]
        return x

    @nn.compact
    def __call__(self, x: jax.Array, z: jax.Array) -> jax.Array:
        for i in range(self.depth + 1):
            dilation = 2 ** (i % self.depth)
            x0 = self.shortcut(x, dilation)
            x = Affine()(x, z)
            x = nn.relu(x)
            x = nn.Conv(
                features=self.ch,
                kernel_size=(self.kernel_size,),
                kernel_dilation=dilation,
                padding="VALID",
                use_bias=False,
            )(x)
            x = x0 + x * self.param(f"alpha_{i}", lambda _, s: jnp.zeros(s), (1,))
        return x


class DilatedDenseNet(nn.Module):
    ch: int
    depth: int
    kernel_size: int
    num_blocks: int
    hidden_dim: int
    stride: int

    @property
    def pad(self) -> int:
        return self.stride * self.num_blocks * (self.kernel_size - 1) * 2 ** self.depth

    @property
    def p(self) -> int:
        return self.pad // 2

    @property
    def dummy_args(self) -> Tuple[jax.Array, jax.Array, jax.Array]:
        dummy_x = jnp.ones((1, self.pad + 1, 2))
        dummy_t = jnp.ones((1,))
        dummy_cond = jnp.ones((1, self.pad + 1, 1))
        return dummy_x, dummy_t, dummy_cond

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        t: jax.Array,
        cond: jax.Array,
    ) -> jax.Array:
        ch_in = x.shape[-1]
        x = jnp.concatenate([x, cond], axis=-1)
        x = nn.Conv(self.ch, kernel_size=(self.stride,), strides=(self.stride,))(x)
        z = nn.relu(nn.Dense(self.hidden_dim)(t.reshape(-1, 1)))
        for _ in range(self.num_blocks):
            x = DilatedBlock(self.ch, self.depth, self.kernel_size)(x, z)
        x = Affine()(x, z)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=ch_in, kernel_size=(self.stride,), strides=(self.stride,))(x)
        return x
