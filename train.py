import jax
import jax.numpy as jnp
import orbax.checkpoint
import optax
import yaml
import hruid

from typing import Tuple, Any, Optional
from typing_extensions import Self

from ml_collections.config_dict import ConfigDict
from flax.training import train_state, orbax_utils

from model import DilatedDenseNet
from data import WaveformDataLoader
from diffusion import diffusion_sampling


class TrainState(train_state.TrainState):

    @classmethod
    def from_config(cls, rng: jax.Array, config: ConfigDict) -> Self:
        net = DilatedDenseNet(**config.model)
        params = net.init(rng, *net.dummy_args)
        tx = get_optimizer(config.optimizer.to_dict(), config.schedule.to_dict())
        state = cls.create(
            apply_fn=net.apply,
            params=params,
            tx=tx,
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
    cond: jax.Array,
) -> Tuple[jax.Array, ...]:

    def compute_loss(params):
        vt_pred = state.apply_fn(params, xt, t, cond)
        cropped_vt = vt[:, state.net.p:-state.net.p]
        loss = jnp.mean((vt_pred - cropped_vt) ** 2)
        return loss

    grad_fn = jax.value_and_grad(compute_loss)
    loss, grads = grad_fn(state.params)
    return grads, loss


@jax.jit
def apply_model_inference(state: TrainState, xt: jax.Array, t: jax.Array, cond: jax.Array) -> jax.Array:
    return state.apply_fn(state.params, xt, t, cond)


@jax.jit
def update_model(state: TrainState, grads: jax.Array) -> TrainState:
    return state.apply_gradients(grads=grads)


def train_step(
    state: TrainState,
    xt: jax.Array,
    t: jax.Array,
    vt: jax.Array,
    cond: jax.Array,
) -> Tuple[TrainState, jax.Array]:
    grads, loss = apply_model_train(state, xt, t, vt, cond)
    state = update_model(state, grads)
    return state, loss


def get_optimizer(opt_params: dict, schedule_params: dict) -> optax.GradientTransformation:
    opt_params = opt_params.copy()
    opt_type = opt_params.pop("type")
    schedule_params = schedule_params.copy()
    schedule_type = schedule_params.pop("type")
    schedule = getattr(optax, schedule_type)(
        peak_value=opt_params.pop("learning_rate"),
        **schedule_params,
    )
    opt_params["learning_rate"] = schedule
    tx = getattr(optax, opt_type)(**opt_params)
    return tx


def count_params(state: TrainState) -> int:
    return sum([jnp.prod(jnp.array(p.shape)) for p in jax.tree_util.tree_flatten(state.params)[0]])


def get_val_len(config: ConfigDict, net_pad: int) -> Tuple[int, int]:
    input_len = config.length or net_pad if config.padded else (
        config.num_steps * net_pad + (config.length or net_pad))
    output_len = input_len if config.padded else input_len - config.num_steps * net_pad
    return input_len, output_len


def save_checkpoint(filepath: str, config: ConfigDict, state: TrainState) -> None:
    ckpt = {"config": config.to_dict(), "state": state}
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(filepath, ckpt, save_args=save_args)


def restore_checkpoint(filepath: str, override_config: Optional[ConfigDict] = None):
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored = checkpointer.restore(filepath)
    config, state = [restored[k] for k in ("config", "state")]
    if config is not None:
        config = ConfigDict(config)
    elif override_config is not None:
        config = override_config
    else:
        raise ValueError("no config found")
    net = DilatedDenseNet(**config.model)
    tx = get_optimizer(config.optimizer.to_dict(), config.schedule.to_dict())
    state = TrainState(
        apply_fn=net.apply,
        tx=tx,
        **state,
    )
    return config, state


def config_from_yaml_str(string):
    return ConfigDict(yaml.safe_load(string))
