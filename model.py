import jax
import jax.numpy as jnp
import flax.linen as nn


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

    def __post_init__(self) -> None:
        self.pad = self.num_blocks * (self.kernel_size - 1) * 2 ** self.depth
        self.p = self.pad // 2
        dummy_x = jnp.ones((1, self.pad + 1, 2))
        dummy_t = jnp.ones((1,), dtype=jnp.int32)
        dummy_cond = jnp.ones((1, self.pad + 1, 1))
        self.dummy_args = (dummy_x, dummy_t, dummy_cond)
        super().__post_init__()

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        t: jax.Array,
        cond: jax.Array,
    ) -> jax.Array:
        ch_in = x.shape[-1]
        x = jnp.concatenate([x, cond], axis=-1)
        x = nn.Conv(self.ch, kernel_size=(1,))(x)
        z = nn.relu(nn.Dense(self.hidden_dim)(t.reshape(-1, 1)))
        for _ in range(self.num_blocks):
            x = DilatedBlock(self.ch, self.depth, self.kernel_size)(x, z)
        x = Affine()(x, z)
        x = nn.relu(x)
        x = nn.Conv(features=ch_in, kernel_size=(1,))(x)
        return x
