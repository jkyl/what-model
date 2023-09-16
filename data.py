import concurrent.futures

import jax
import jax.numpy as jnp
import flax.linen as nn
import soundfile as sf

from functools import partial
from typing_extensions import Self
from queue import Queue, Empty, Full
from dataclasses import dataclass
from typing import Optional, Tuple, Union

from ml_collections.config_dict import ConfigDict
from scipy.datasets import electrocardiogram as ecg

from diffusion import compose_diffusion_batch


def stereo_to_mid_side(x: jax.Array) -> jax.Array:
    assert x.ndim == 2 and x.shape[1] == 2, x.shape
    # Reflection about y=x.
    M = (0.5 ** 0.5) * jnp.array([[1., 1.], [1., -1.]])
    return x @ M


# It's its own inverse (orthonormal).
mid_side_to_stereo = stereo_to_mid_side


@partial(jax.jit, static_argnums=(2, 3, 4))
def batch_crops(data: jax.Array, starts: jax.Array, batch_size: int, length: int, p: int) -> jax.Array:
    indices = starts[:, None] + jnp.arange(length)
    batch = data[indices]
    if p > 0:
        mask = jnp.logical_or(indices < p, indices >= data.shape[0] - p)
        return batch, mask[..., None]
    return batch


class WaveformDataset(nn.Module):
    filename: Optional[str] = None
    mono: bool = False
    p: int = 0

    def _init(self, _) -> jax.Array:
        if self.filename is not None:
            data, _ = sf.read(self.filename)
            if data.ndim == 2 and self.mono:
                data = stereo_to_mid_side(data)[:, :1]
            if data.ndim == 1:
                data = data[:, None]
        else:
            data = ecg()[:, None]
            
        data /= data.std()
        data = jnp.array(data, dtype=jnp.float32)
        data = jnp.pad(data, ((self.p, self.p), (0, 0)))
        return data

    @nn.compact
    def __call__(self, batch_size: int, length: int) -> Union[jax.Array, Tuple[jax.Array, ...]]:
        rng = self.make_rng("crops")
        data = self.param("data", self._init)
        starts = jax.random.randint(rng, (batch_size,), 0, data.shape[0] - length)
        return batch_crops(data, starts, batch_size, length, self.p)


@dataclass
class WaveformSampler:
    dataset: WaveformDataset
    batch_size: int
    length: int
    
    @jax.default_device(jax.devices("cpu")[0])
    def __post_init__(self):
        dummy_rng = jax.random.PRNGKey(0)  # The parameters of this module are static.
        self.params = self.dataset.init({"params": dummy_rng, "crops": dummy_rng}, self.batch_size, self.length)

    @jax.default_device(jax.devices("cpu")[0])
    def __getitem__(self, rng: jax.Array):
        return self.dataset.apply(self.params, self.batch_size, self.length, rngs={"crops": rng})


@dataclass
class WaveformDataLoader:
    rng: jax.Array
    datagen: WaveformSampler
    num_threads: int
    max_queue_length: int

    @property
    def _shutdown(self):
        return "SHUTDOWN"

    def __post_init__(self):
        self.data_queue = Queue(self.max_queue_length)
        self.control_queue = Queue(maxsize=self.num_threads)
        self.worker_exceptions = []

    @jax.default_device(jax.devices("cpu")[0])
    def _worker(self, rng_key):
        batch = None
        try:
            while True:
                try:
                    command = self.control_queue.get_nowait()
                    if command == self._shutdown:
                        break
                except Empty:
                    pass

                if batch is None:
                    rng_key, *batch = compose_diffusion_batch(rng_key, self.datagen)

                try:
                    self.data_queue.put(batch, timeout=0.5)
                    batch = None
                except Full:
                    pass
        except Exception as e:
            print("exception in worker", e)
            self.worker_exceptions.append(e)

    def __enter__(self):
        rng_keys = jax.random.split(self.rng, self.num_threads)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads)
        self.futures = [self.executor.submit(self._worker, rng_key) for rng_key in rng_keys]
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for _ in range(self.num_threads):
            self.control_queue.put(self._shutdown)
        self.executor.shutdown()

    def get(self):
        if self.worker_exceptions:
            raise RuntimeError("Worker thread exceptions detected.") from self.worker_exceptions[0]
        batch = self.data_queue.get()
        gpu_if_avail = jax.devices()[0]
        batch = [jax.device_put(elem, gpu_if_avail) for elem in batch]
        return batch

    @classmethod
    def from_config(cls, rng, config: ConfigDict, fallbacks: dict) -> Self:
        dataset = WaveformDataset(
            filename=config.filename, 
            mono=config.mono, 
            p=config.p or fallbacks["p"],
        )
        datagen = WaveformSampler(
            dataset,
            config.batch_size,
            config.length or fallbacks["length"],
        )
        return cls(
            rng, 
            datagen, 
            num_threads=config.num_threads, 
            max_queue_length=config.max_queue_length,
        )
