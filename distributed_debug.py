import dataclasses
import logging
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import rich.logging
import yaml
from jax._src.distributed import initialize as jax_distributed_initialize

from purejaxql.config import get_config

logger = logging.getLogger(__name__)

SCRATCH = Path(os.environ["SCRATCH"])


def get_gpus_per_task(tres_per_task: str) -> int:
    """Returns the number of GPUS per task from the SLURM env variables.

    >>> get_gpus_per_task('cpu=12,gres/gpu=1')
    1
    >>> get_gpus_per_task('cpu=48,gres/gpu=4')
    4
    >>> get_gpus_per_task('cpu=48,gres/gpu=h100=2')
    2
    >>> get_gpus_per_task('cpu=48')
    0
    >>> get_gpus_per_task('cpu=48,gres/gpu:h100=4')
    """
    # todo: figure out how many GPUS per task given the tres_per_task.
    # Example: 'cpu=48,gres/gpu=4' --> 4
    # Example: 'cpu=48,gres/gpu=h100:4' --> 4
    for part in tres_per_task.split(","):
        res_type, _, res_count = part.partition("=")
        if res_type == "gres/gpu":
            gpus_per_task = int(res_count.rpartition(":")[-1])

            assert gpus_per_task > 0
            return gpus_per_task
    raise NotImplementedError(f"Unknown tres_per_task format: {tres_per_task}")


@dataclasses.dataclass(frozen=True)
class SlurmDistributedEnv:
    """Distributed training context derived from SLURM Environment variables."""

    job_id: int = dataclasses.field(
        default_factory=lambda: int(os.environ["SLURM_JOB_ID"])
    )
    global_rank: int = dataclasses.field(
        default_factory=lambda: int(os.environ["SLURM_PROCID"])
    )
    local_rank: int = dataclasses.field(
        default_factory=lambda: int(os.environ["SLURM_LOCALID"])
    )
    num_tasks: int = dataclasses.field(
        default_factory=lambda: int(os.environ["SLURM_NTASKS"])
    )
    num_nodes: int = dataclasses.field(
        default_factory=lambda: int(os.environ["SLURM_JOB_NUM_NODES"])
    )
    ntasks_per_node: int = dataclasses.field(
        default_factory=lambda: int(os.environ["SLURM_NTASKS_PER_NODE"])
    )
    cpus_per_task: int = dataclasses.field(
        default_factory=lambda: int(os.environ["SLURM_CPUS_PER_TASK"])
    )
    gpus_per_task: int = dataclasses.field(
        default_factory=lambda: get_gpus_per_task(os.environ["SLURM_TRES_PER_TASK"])
        or int(os.environ["SLURM_GPUS_ON_NODE"])
    )
    node_id: int = dataclasses.field(
        default_factory=lambda: int(os.environ["SLURM_NODEID"])
    )
    node_list: tuple[str, ...] = dataclasses.field(
        default_factory=lambda: tuple(os.environ["SLURM_JOB_NODELIST"].split(","))
    )


def setup_logging(local_rank: int, num_processes: int, verbose: int):
    logging.basicConfig(
        level=logging.INFO,
        # Add the [{local_rank}/{num_processes}] prefix to log messages
        format=(
            (f"[{local_rank + 1}/{num_processes}] " if num_processes > 1 else "")
            + "%(message)s"
        ),
        handlers=[
            rich.logging.RichHandler(show_time=False, rich_tracebacks=True, markup=True)
        ],
        force=True,
    )
    if verbose == 0:
        logger.setLevel(logging.ERROR)
    elif verbose == 1:
        logger.setLevel(logging.WARNING)
    elif verbose == 2:
        logger.setLevel(logging.INFO)
    else:
        assert verbose >= 3
        logger.setLevel(logging.DEBUG)


from jax.experimental.shard_map import shard_map  # noqa
from jax.sharding import Mesh, NamedSharding, PartitionSpec  # noqa
from jax.sharding import PartitionSpec as P


def loss(params, batch):
    inputs, targets = batch
    predictions = predict(params, inputs)
    return jnp.mean(jnp.sum((predictions - targets) ** 2, axis=-1))


def init_layer(key, n_in, n_out):
    k1, k2 = jax.random.split(key)
    W = jax.random.normal(k1, (n_in, n_out)) / jnp.sqrt(n_in)
    b = jax.random.normal(k2, (n_out,))
    return W, b


def init(key, layer_sizes, batch_size):
    key, *keys = jax.random.split(key, len(layer_sizes))
    params = list(map(init_layer, keys, layer_sizes[:-1], layer_sizes[1:]))

    key, *keys = jax.random.split(key, 3)
    inputs = jax.random.normal(keys[0], (batch_size, layer_sizes[0]))
    targets = jax.random.normal(keys[1], (batch_size, layer_sizes[-1]))

    return params, (inputs, targets)


def predict(params: list[tuple[jax.Array, jax.Array]], inputs: jax.Array) -> jax.Array:
    outputs = None
    for W, b in params:
        outputs = jnp.dot(inputs, W) + b
        inputs = jax.nn.relu(outputs)
    assert outputs is not None
    return outputs


def main():
    dist_env = SlurmDistributedEnv()
    setup_logging(dist_env.global_rank, dist_env.num_tasks, 2)
    config = get_config()
    if dist_env.global_rank == 0:
        print("Distributed env:\n", yaml.dump(dataclasses.asdict(dist_env), indent=2))
        print("Config:\n", yaml.dump(dataclasses.asdict(config), indent=2))

    task_gpus = list(range(dist_env.gpus_per_task))
    # task_gpus = list(
    #     range(
    #         dist_env.local_rank * dist_env.gpus_per_task,
    #         (dist_env.local_rank + 1) * dist_env.gpus_per_task,
    #     )
    # )
    # # todo: adjust if we want to do one task per gpu.
    jax_distributed_initialize(
        local_device_ids=task_gpus, num_processes=dist_env.num_tasks
    )
    logger.info(f"{jax.devices()=}, {jax.local_devices()=}")

    mesh = jax.make_mesh((jax.device_count(),), ("batch",))

    layer_sizes = [784, 128, 128, 128, 128, 128, 8]
    batch_size = 32

    params, batch = init(jax.random.key(0), layer_sizes, batch_size)

    # replicate initial params on all devices, shard data batch over devices
    batch = jax.device_put(batch, NamedSharding(mesh, P("batch")))
    params = jax.device_put(params, NamedSharding(mesh, P()))

    # adapt the loss function to sum the losses across devices
    def loss_fn(
        local_batch: tuple[jax.Array, jax.Array],
        params: list[tuple[jax.Array, jax.Array]],
    ):
        inputs, targets = local_batch
        predictions = predict(params, inputs)  # use reference 'predict`
        # NOTE: Keeping the same exact loss function here doesn't work!
        # loss = jnp.mean(jnp.sum((predictions - targets) ** 2, axis=-1))
        # return loss
        local_loss = jnp.mean(jnp.sum((predictions - targets) ** 2, axis=-1))
        return jax.lax.pmean(local_loss, "batch")

    def loss_dp(params, batch):
        # @partial(shard_map, mesh=mesh, in_specs=P("batch", None), out_specs=P())
        # def loss_spmd(local_batch):
        loss_spmd = shard_map(
            loss_fn,
            mesh=mesh,
            in_specs=(P("batch"), P()),
            out_specs=P(),
        )
        return loss_spmd(batch, params)
        # return jax.lax.pmean(loss_spmd(batch, params), "batch")

    print(jax.jit(loss)(params, batch))
    print(jax.jit(loss_dp)(params, batch))


if __name__ == "__main__":
    main()
