import jax
import jax.numpy as jnp
import tqdm

from typing import Tuple, Callable, Mapping, Optional, Union


def alpha(t: jax.Array) -> jax.Array:
    return (jnp.cos(jnp.pi * t) + 1) / 2.


def x0_and_e_to_xt_and_vt(x0: jax.Array, e: jax.Array, at: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    https://arxiv.org/pdf/2202.00512.pdf#page=14
    """
    a, s = alpha_sigma(at)
    xt = a * x0 + s * e
    vt = a * e - s * x0
    return xt, vt


def vt_and_xt_to_x0_and_e(vt: jax.Array, xt: jax.Array, at: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    https://arxiv.org/pdf/2202.00512.pdf#page=14
    """
    a, s = alpha_sigma(at)
    x0 = a * xt - s * vt
    e = jnp.where(a == 0, xt, (vt + s * x0) / a)  # Zero terminal SNR <=> input is pure noise at t=T.
    return x0, e


def compose_diffusion_batch(rng: jax.Array, datagen: Mapping) -> Tuple[jax.Array, ...]:
    rng, x0_key = jax.random.split(rng)
    x0 = datagen[x0_key]
    rng, e_key = jax.random.split(rng)
    e = jax.random.normal(e_key, x0.shape)
    rng, t_key = jax.random.split(rng)
    t = jax.random.uniform(t_key, (x0.shape[0],))
    at = alpha(t).reshape(-1, 1, 1)
    xt, vt = x0_and_e_to_xt_and_vt(x0, e, at)
    return rng, xt, t, vt


def get_timesteps(num_steps: int, batch_size: int) -> jax.Array:
    num_steps += 1
    timesteps = jnp.linspace(0, 1, num_steps)
    next_timesteps = jnp.roll(timesteps, 1)
    timesteps = jnp.stack([timesteps, next_timesteps], axis=1)
    timesteps = timesteps[::-1]  # Diffusion goes from t=T to t=0.
    batches = jnp.repeat(timesteps.reshape(num_steps, 2, 1), batch_size, axis=2)
    return batches[:-1]  # Don't need the last step for generation.


def alpha_sigma(a: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    Breaks alpha(t) into alpha and sigma, shorthand for
    their variance-preserving complimentary coefficients.
    """
    return a ** 0.5, (1 - a) ** 0.5


def ddim_sampling_step(
    *,
    rng: jax.Array,
    model: Callable[[jax.Array, jax.Array], jax.Array],
    t: jax.Array,
    t_next: jax.Array,
    eta: float = 0.,
    xt: jax.Array
) -> Tuple[jax.Array, jax.Array]:

    at = alpha(t).reshape(-1, 1, 1)
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
    timesteps: Optional[jax.Array] = None,
    progbar: Union[bool, tqdm.tqdm] = True,
) -> jax.Array:

    # Base case.
    if num_steps == 0:
        return xt

    if progbar is True:
        progbar = tqdm.tqdm(total=num_steps)

    if timesteps is None:
        timesteps = get_timesteps(num_steps, xt.shape[0])

    t, t_next = timesteps[-num_steps]

    # Execute a step.
    rng, xt = step_fun(
        rng=rng,
        model=model,
        t=t,
        t_next=t_next,
        eta=eta,
        xt=xt
    )

    if progbar is not False:
        progbar.update()

    # Recursive call.
    return diffusion_sampling(
        rng=rng,
        model=model,
        xt=xt,
        num_steps=num_steps - 1,
        eta=eta,
        timesteps=timesteps,
        progbar=progbar,
    )
