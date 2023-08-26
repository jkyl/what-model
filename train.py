import jax
import jax.numpy as jnp
import optax
import hydra

from typing import Tuple, Any
from typing_extensions import Self

from omegaconf import DictConfig
from flax.training import train_state

from model import DilatedDenseNet
from data import WaveformDataLoader
from diffusion import diffusion_sampling


class TrainState(train_state.TrainState):
    batch_stats: Any

    @classmethod
    def from_config(cls, rng: jax.Array, config: DictConfig) -> Self:
        net = DilatedDenseNet(**config.model)
        variables = net.init(rng, *net.dummy_args, train=True)
        params = variables["params"]
        batch_stats = variables["batch_stats"]
        tx = get_optimizer(**config.optimizer)
        state = cls.create(
            apply_fn=net.apply,
            params=params,
            tx=tx,
            batch_stats=batch_stats,
        )
        return state

    @property
    def net(self):
        return self.apply_fn.__self__


@jax.jit
def apply_model_train(
    state: TrainState,
    xt: jax.Array,
    t: jax.Array,
    vt: jax.Array,
) -> Tuple[jax.Array, ...]:

    def compute_loss(params):
        vt_pred, updates = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            xt, t,
            train=True,
            mutable=["batch_stats"],
        )
        cropped_vt = vt[:, state.net.p:-state.net.p]
        loss = jnp.mean((vt_pred - cropped_vt) ** 2)
        return loss, (vt_pred, updates)

    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
    (loss, (vt_pred, updates)), grads = grad_fn(state.params)
    return grads, loss, vt_pred, updates


@jax.jit
def apply_model_inference(state: TrainState, xt: jax.Array, t: jax.Array) -> jax.Array:
    return state.apply_fn({"params": state.params, "batch_stats": state.batch_stats}, xt, t, train=False)


@jax.jit
def update_model(state: TrainState, grads: jax.Array, updates: dict) -> TrainState:
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates["batch_stats"])
    return state


def train_step(
    state: TrainState,
    xt: jax.Array,
    t: jax.Array,
    v: jax.Array,
) -> Tuple[TrainState, jax.Array]:
    grads, loss, e_pred, updates = apply_model_train(state, xt, t, v)
    state = update_model(state, grads, updates)
    return state, loss


def get_optimizer(**kwargs) -> optax.GradientTransformation:
    opt_params = kwargs.copy()
    opt_type = opt_params.pop("type")
    tx = getattr(optax, opt_type)(**opt_params)
    return tx


def count_params(state: TrainState) -> int:
    return sum([jnp.prod(jnp.array(p.shape)) for p in jax.tree_util.tree_flatten(state.params)[0]])


def interval(step: int, rate: int) -> bool:
    return step and ((step + 1) % rate == 0)


def get_val_len(config: DictConfig, net_pad: int) -> Tuple[int, int]:
    input_len = config.length or net_pad if config.padded else (
        config.num_steps * net_pad + (config.length or net_pad))
    output_len = input_len if config.padded else input_len - config.num_steps * net_pad
    return input_len, output_len


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config: DictConfig):
    init_rng = jax.random.PRNGKey(config.rngs.init)
    state = TrainState.from_config(init_rng, config)
    print("params:", count_params(state))
    data_rng = jax.random.PRNGKey(config.rngs.data)
    data = WaveformDataLoader.from_config(data_rng, config.data, p=state.net.p, length=2*state.net.pad)
    assert data.datagen.length > state.net.pad
    val_rng = jax.random.PRNGKey(config.rngs.val)
    val_input_len, val_output_len = get_val_len(config.validation, state.net.pad)
    val_xt = jax.random.normal(val_rng, (1, val_input_len, 1))

    import matplotlib.pyplot as plt
    x = 2 * jax.random.normal(val_rng, (val_output_len,))
    fig, ax = plt.subplots()
    line, = ax.plot(x)
    plt.ion()
    plt.show()

    with data:
        losses = []
        for step in range(config.training.num_iters):
            state, loss = train_step(state, *data.get())
            losses.append(loss)
            if interval(step, config.training.log_interval):
                print(
                    f"steps={step + 1}, avg_loss="
                    f"{jnp.mean(jnp.array(losses[-config.training.log_interval:])):.3f}")
            if interval(step, config.validation.interval):
                print("generating")
                x0_pred = diffusion_sampling(
                    rng=val_rng,
                    model=lambda xt, t: apply_model_inference(state, xt, t),
                    xt=val_xt,
                    num_steps=config.validation.num_steps,
                    p=state.net.p if config.validation.padded else 0,
                )
                line.set_ydata(x0_pred.squeeze())
                plt.draw()
                plt.pause(0.5)


if __name__ == "__main__":
    main()
