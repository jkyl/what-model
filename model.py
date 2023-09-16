import jax
import jax.numpy as jnp
import flax.linen as nn

from functools import partial
from typing import Optional


def concat_ragged(*args: jax.Array):
    lengths = [a.shape[1] for a in args]
    min_length = min(lengths)
    cropped = []
    for arr, length in zip(args, lengths):
        crop = (length - min_length) // 2
        cropped.append(arr[:, crop:-crop] if crop else arr)
    catted = jnp.concatenate(cropped, axis=2)
    return catted


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
            x = x0 + x * self.param(f"alpha_{i}", lambda _, I: jnp.zeros(I), (1,))
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
        cat = [nn.Conv(self.ch, kernel_size=(1,))(x)]
        z = nn.relu(nn.Dense(self.hidden_dim)(t.reshape(-1, 1)))
        for _ in range(self.num_blocks):
            cat.append(DilatedBlock(self.ch, self.depth, self.kernel_size)(cat[-1], z))
        x = concat_ragged(*cat)
        x = Affine()(x, z)
        x = nn.relu(x)
        x = nn.Conv(features=ch_in, kernel_size=(1,))(x)
        return x


@partial(jax.jit, static_argnums=1)
def space_to_depth(x: jax.Array, factor: int = 2):
    n, l, c = x.shape
    return x.reshape(n, l // 2, 2 * c)


@partial(jax.jit, static_argnums=1)
def depth_to_space(x: jax.Array, factor: int = 2):
    n, l, c = x.shape
    return x.reshape(n, 2 * l, c // 2)
    
    
class UNet(nn.Module):
    ch: int
    depth: int
    hidden_dim: int
    num_heads: int
    
    def __post_init__(self) -> None:
        self.pad = self.p = 0
        dummy_xt = jnp.ones((1, 2 ** (self.depth + 2), 1))
        dummy_t = jnp.ones((1,), dtype=jnp.int32)
        self.dummy_args = (dummy_xt, dummy_t)
        self.z_dim = 4 * self.ch * self.depth
        super().__post_init__()
    
    @nn.compact
    def __call__(self, x: jax.Array, t: jax.Array, *, train: bool) -> jax.Array:
        ch_in = x.shape[-1]
        x = nn.Conv(self.ch, kernel_size=(1,), use_bias=False)(x)
        z = MLP(self.hidden_dim, self.z_dim)(t.reshape(-1, 1))
        skip = []
        pointer = 0
        for i in range(self.depth):
            dim = x.shape[-1]
            scale, bias = jnp.split(z[:, pointer : (pointer := pointer + 2 * dim)], 2, axis=1)
            x = ConditionalBatchNorm()(x, scale, bias, train=train)
            x = nn.relu(x)
            x = nn.Conv(dim, kernel_size=(1,), use_bias=False)(x)
            skip.append(x[..., :dim//2])
            x = space_to_depth(x[..., dim//2:], 2)
        x = RelativeSelfAttention(self.num_heads)(x)
        for i in reversed(range(self.depth)):
            x = depth_to_space(x)
            x = jnp.concatenate([skip.pop(-1), x], axis=2)
            dim = x.shape[-1]
            scale, bias = jnp.split(z[:, pointer : (pointer := pointer + 2 * dim)], 2, axis=1)
            x = ConditionalBatchNorm()(x, scale, bias, train=train)
            x = nn.relu(x)
            x = nn.Conv(dim, kernel_size=(1,), use_bias=False)(x)
        x = ConditionalBatchNorm()(x, train=train)
        x = nn.relu(x)
        x = nn.Conv(ch_in, kernel_size=(1,))(x)
        return x


def sinusoidal_embeddings(length: int, dim: int, decay_term: float = 1e4) -> jax.Array:
    positions = jnp.arange(1 - length, length).reshape(-1, 1)
    div_term = jnp.exp(-jnp.arange(0, dim, 2) * (jnp.log(decay_term) / dim))
    sin_term = jnp.sin(positions * div_term)
    cos_term = jnp.cos(positions * div_term)
    pos_enc = jnp.concatenate([sin_term, cos_term], axis=1)
    return pos_enc


def compute_relative_scores(
    query: jax.Array,
    embeddings_matrix: jax.Array,
    use_optimized_impl: bool,
) -> jax.Array:

    batch_size, num_heads, length, dim = query.shape
    assert embeddings_matrix.shape == (2 * length - 1, dim)

    if use_optimized_impl:
        return _compute_relative_scores_with_conv(query, embeddings_matrix)

    return _compute_relative_scores_with_indexing(query, embeddings_matrix)


def _compute_relative_scores_with_indexing(query: jax.Array, embeddings_matrix: jax.Array) -> jax.Array:
    batch_size, num_heads, length, dim = query.shape
    indices = (jnp.arange(length)[:, None] - jnp.arange(length)) + length - 1
    embeddings = embeddings_matrix[indices]
    qe = jnp.einsum("bhqd,qkd->bhqk", query, embeddings)
    return qe


def _compute_relative_scores_with_conv(query: jax.Array, embeddings_matrix: jax.Array) -> jax.Array:
    raise NotImplementedError


class RelativeSelfAttention(nn.Module):
    num_heads: int
    optimized: bool = False

    def _split_heads(self, x: jax.Array) -> jax.Array:
        """Split channels (axis 2) into multiple heads along a new axis (position 1)."""
        return jnp.split(x.reshape(x.shape[0], x.shape[1], self.num_heads, -1).transpose(0, 2, 1, 3), 3, axis=3)

    @staticmethod
    def _combine_heads(x: jax.Array) -> jax.Array:
        """Combine multi-head outputs into original-shaped array."""
        return x.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], -1)

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:

        # Check that the input shape is valid.
        _, length, dim = x.shape
        assert dim % self.num_heads == 0, "dim must be divisible by num_heads."
        d_head = dim // self.num_heads

        # Project input onto queries, keys, and values and split amongst heads.
        q, k, v = self._split_heads(nn.Conv(3 * dim, kernel_size=(1,))(x))

        # Correlate queries and keys.
        qk = jnp.einsum("bhqd,bhkd->bhqk", q, k)

        # Generate sinusoidal embeddings.
        embeddings_matrix = sinusoidal_embeddings(length, d_head)

        # Correlate queries and embeddings.
        qe = compute_relative_scores(q, embeddings_matrix, use_optimized_impl=self.optimized)

        # Combine scores to form logits.
        logits = qk + qe

        # Normalize logits.
        scaled_logits = logits / jnp.sqrt(d_head)
        weights = jax.nn.softmax(scaled_logits)

        # Reduce across keys.
        output = jnp.einsum("bhqk,bhkd->bhqd", weights, v)

        # Collapse heads axis back to channels.
        return self._combine_heads(output)
