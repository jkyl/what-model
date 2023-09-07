import jax
import jax.numpy as jnp
import tqdm

from typing import Tuple, Callable, Mapping, Optional


def alpha_sigma(t: jax.Array) -> Tuple[jax.Array, jax.Array]:
    t = t.reshape(-1, 1, 1)
    alpha = jnp.cos(0.5 * jnp.pi * t)
    sigma = jnp.sqrt(1 - alpha ** 2)
    return alpha, sigma


def x0_and_e_to_xt_and_vt(x0: jax.Array, e: jax.Array, t: jax.Array) -> Tuple[jax.Array, jax.Array]:
    a, s = alpha_sigma(t)
    xt = a * x0 + s * e
    vt = a * e - s * x0
    return xt, vt


def vt_and_xt_to_x0_and_e(vt: jax.Array, xt: jax.Array, t: jax.Array) -> Tuple[jax.Array, jax.Array]:
    a, s = alpha_sigma(t)
    x0 = a * xt - s * vt
    e = jnp.where(a == 0, xt, (vt + s * x0) / a)  # Zero terminal SNR <=> input is pure noise at t=T.
    return x0, e


def compose_diffusion_batch(rng: jax.Array, datagen: Mapping) -> Tuple[jax.Array, ...]:
    rng, x0_key = jax.random.split(rng)
    x0, mask = datagen[x0_key]  # Maybe zero-padded.
    cond = mask.astype(jnp.float32)  # Condition the model on the mask: OOB indices are 1., in-bound are 0. 
    # We may consider adding other conditioning -- for example sinusoidal embeddings, or elapsed seconds. 
    rng, e_key = jax.random.split(rng)
    e = jax.random.normal(e_key, x0.shape)
    rng, t_key = jax.random.split(rng)
    t = jax.random.uniform(t_key, (x0.shape[0],))
    xt, vt = x0_and_e_to_xt_and_vt(x0, e, t)
    return rng, xt, t, vt, cond


def get_timesteps(num_steps: int, batch_size: int) -> jax.Array:
    timesteps = jnp.linspace(0, 1, num_steps + 1)  # Inclusive of lower and upper bounds, so +1. 
    next_timesteps = jnp.roll(timesteps, 1)
    timesteps = jnp.stack([timesteps, next_timesteps], axis=1)
    timesteps = timesteps[::-1]  # Diffusion goes from t=T to t=0.
    batches = jnp.repeat(timesteps.reshape(-1, 2, 1), batch_size, axis=2)
    return batches[:-1]  # Don't need the last step for generation.


def ddim_sampling_step(
    *,
    rng: jax.Array,
    model: Callable[[jax.Array, jax.Array], jax.Array],
    xt: jax.Array,
    t: jax.Array,
    t_next: jax.Array,
    eta: float = 0.
) -> Tuple[jax.Array, jax.Array]:
    vt = model(xt, t)
    p = (xt.shape[1] - vt.shape[1]) // 2
    xt = xt[:, p:-p] if p else xt
    x0, e = vt_and_xt_to_x0_and_e(vt, xt, t)
    at_next, st_next = alpha_sigma(t_next)
    c1 = eta * st_next / at_next
    c2 = jnp.sqrt(1 - at_next ** 2 - c1 ** 2)
    rng, key = jax.random.split(rng)
    xt_next = at_next * x0 + c1 * jax.random.normal(key, x0.shape) + c2 * e
    return rng, xt_next


def diffusion_sampling(
    *,
    rng: jax.Array,
    model: Callable[[jax.Array, jax.Array], jax.Array],
    xt: jax.Array,
    num_steps: int,
    padded: bool = True,
    progbar: Optional[bool] = None,
) -> jax.Array:

    xT = xt  # x at t=T is pure noise. use this to evolve the zero-padded regions, which are 0 at t=0.
    timesteps = get_timesteps(num_steps, xt.shape[0])
    for t, t_next in tqdm.tqdm(timesteps, disable=(not progbar)):
        rng, xt = ddim_sampling_step(rng=rng, model=model, xt=xt, t=t, t_next=t_next)
        p = (xT.shape[1] - xt.shape[1]) // 2
        if p and padded:
            xt_pad, _ = x0_and_e_to_xt_and_vt(jnp.zeros(xT.shape), xT, t_next)
            xt = xt_pad.at[:, p:-p].set(xt)
    return xt
