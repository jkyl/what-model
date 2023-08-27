import concurrent.futures

import jax
import jax.numpy as jnp
import flax.linen as nn
import soundfile as sf

from typing_extensions import Self
from queue import Queue, Empty, Full
from dataclasses import dataclass

from typing import Optional
from omegaconf import DictConfig
from scipy.datasets import electrocardiogram as ecg

from common import batch_crops
from diffusion import compose_diffusion_batch


class WaveformDataset(nn.Module):
    filename: Optional[str]

    @staticmethod
    def _init(_, filename: str) -> jax.Array:
        if filename is not None:
            data, _ = sf.read(filename)
            data = data.mean(axis=1)
        else:
            data = ecg()
        data -= data.mean()
        data /= data.std()
        return jnp.array(data, dtype=jnp.float32)

    @nn.compact
    def __call__(self, batch_size: int, length: int, p: int = 0):
        rng = self.make_rng("crops")
        data = self.param("data", self._init, self.filename)
        starts = jax.random.randint(rng, (batch_size,), -p, data.size - length + p)
        return batch_crops(data, starts, length)[..., None]


@dataclass
class WaveformSampler:
    dataset: WaveformDataset
    batch_size: int
    length: int
    p: int = 0

    def __post_init__(self):
        dummy_rng = jax.random.PRNGKey(0)  # The parameters of this module are static.
        self.params = self.dataset.init({"params": dummy_rng, "crops": dummy_rng}, self.batch_size, self.length)

    def __getitem__(self, rng: jax.Array):
        return self.dataset.apply(self.params, self.batch_size, self.length, self.p, rngs={"crops": rng})


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

    @jax.default_device(jax.devices("cpu")[0])
    def _worker(self, rng_key):
        batch = None
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
        return self.data_queue.get()

    @classmethod
    def from_config(cls, rng, config: DictConfig, **fallbacks) -> Self:
        dataset = WaveformDataset(filename=config.filename)
        datagen = WaveformSampler(
            dataset,
            config.batch_size,
            config.length or fallbacks["length"],
            config.p or fallbacks["p"],
        )
        return cls(rng, datagen, config.num_threads, config.max_queue_length)
