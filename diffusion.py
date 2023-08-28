import jax
import jax.numpy as jnp
import tqdm

from typing import Tuple, Callable, Mapping, Optional


def alpha(t: jax.Array) -> jax.Array:
    """
    Defines the diffusion schedule as a function of t ∈ [0, 1].
    """
    return (jnp.cos(jnp.pi * t.reshape(-1, 1, 1)) + 1) / 2.


def alpha_sigma(a: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    Breaks alpha(t) into alpha and sigma, shorthand for
    their variance-preserving complimentary coefficients.
    """
    return a ** 0.5, (1 - a) ** 0.5


def x0_and_e_to_xt_and_vt(x0: jax.Array, e: jax.Array, at: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    Converts initial values and noise to their equivalents at a specific time t.
    """
    a, s = alpha_sigma(at)
    xt = a * x0 + s * e
    vt = a * e - s * x0
    return xt, vt


def vt_and_xt_to_x0_and_e(vt: jax.Array, xt: jax.Array, at: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    Reverts evolved states and noises back to their initial forms.
    """
    a, s = alpha_sigma(at)
    x0 = a * xt - s * vt
    e = jnp.where(a == 0, xt, (vt + s * x0) / a)  # Zero terminal SNR <=> input is pure noise at t=T.
    return x0, e


def compose_diffusion_batch(rng: jax.Array, datagen: Mapping) -> Tuple[jax.Array, ...]:
    """
    Generates initial states, noise, and diffusion schedule for a batch.
    """
    rng, x0_key = jax.random.split(rng)
    x0, mask = datagen[x0_key]  # Maybe zero-padded.
    rng, e_key = jax.random.split(rng)
    e = jax.random.normal(e_key, x0.shape)
    e = jnp.where(mask, 0, e)  # OOB regions are zero.
    rng, t_key = jax.random.split(rng)
    t = jax.random.uniform(t_key, (x0.shape[0],))
    at = alpha(t)
    xt, vt = x0_and_e_to_xt_and_vt(x0, e, at)
    return rng, xt, t, vt


def get_timesteps(num_steps: int, batch_size: int) -> jax.Array:
    """
    Creates a tensor of time steps for the diffusion process.
    """
    num_steps += 1
    timesteps = jnp.linspace(0, 1, num_steps)
    next_timesteps = jnp.roll(timesteps, 1)
    timesteps = jnp.stack([timesteps, next_timesteps], axis=1)
    timesteps = timesteps[::-1]  # Diffusion goes from t=T to t=0.
    batches = jnp.repeat(timesteps.reshape(num_steps, 2, 1), batch_size, axis=2)
    return batches[:-1]  # Don't need the last step for generation.


def ddim_sampling_step(
    *,
    rng: jax.Array,
    model: Callable[[jax.Array, jax.Array], jax.Array],
    t: jax.Array,
    t_next: jax.Array,
    eta: float = 0.,
    xt: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """
    Executes a single DDIM sampling step.
    """
    at = alpha(t)
    at_next = alpha(t_next)
    vt = model(xt, t)
    p = (xt.shape[1] - vt.shape[1]) // 2
    xt = xt[:, p:-p] if p else xt
    x0, e = vt_and_xt_to_x0_and_e(vt, xt, at)
    c1 = eta * jnp.sqrt((1 - at / at_next) * (1 - at_next) / (1 - at))
    c2 = jnp.sqrt(1 - at_next - c1 ** 2)
    rng, key = jax.random.split(rng)
    xt_next = jnp.sqrt(at_next) * x0 + c1 * jax.random.normal(key, x0.shape) + c2 * e
    return rng, xt_next


def diffusion_sampling(
    *,
    rng: jax.Array,
    model: Callable[[jax.Array, jax.Array], jax.Array],
    xt: jax.Array,
    num_steps: int,
    eta: float = 0.,
    step_fun: Callable = ddim_sampling_step,
    progbar: Optional[bool] = True,
    p: int = 0,
) -> jax.Array:
    """
    Performs diffusion sampling across multiple steps.
    """
    timesteps = get_timesteps(num_steps, xt.shape[0])
    for step in tqdm.tqdm(reversed(range(num_steps)), disable=(not progbar)):
        xt = jax.lax.pad(xt, 0., [(0, 0, 0), (p, p, 0), (0, 0, 0)]) if p else xt
        t, t_next = timesteps[-step-1]
        rng, xt = step_fun(
            rng=rng,
            model=model,
            t=t,
            t_next=t_next,
            eta=eta,
            xt=xt
        )
    return xt
