"""
This script uses BatchRenorm for more effective batch normalization in long training runs.

```
JAX_TRACEBACK_FILTERING=off srun --pty --nodes=1 --ntasks-per-node=1 --gpus-per-task=4 --cpus-per-task=48 --mem=0 --partition=gpubase_bynode_b3 \
    uv run python purejaxql/pqn_rnn_craftax_scaled.py +alg=pqn_rnn_craftax alg.TOTAL_TIMESTEPS=100 alg.TOTAL_TIMESTEPS_DECAY=100 NUM_SEEDS=4
```
"""

import copy
import dataclasses
import functools
import logging
import os
import time
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

import chex
import flax.jax_utils
import flax.linen as nn
import flax.struct
import hydra
import jax
import jax.distributed
import jax.experimental
import jax.experimental.mesh_utils
import jax.numpy as jnp
import numpy as np
import optax
import rich.logging
from craftax.craftax.craftax_state import EnvParams, EnvState
from craftax.craftax_env import make_craftax_env_from_name
from craftax_wrappers import (
    BatchEnvWrapper,
    GymnaxWrapper,
    LogWrapper,
    OptimisticResetVecEnvWrapper,
)
from flax.core.frozen_dict import FrozenDict
from flax.linen.module import Module, compact, merge_param
from flax.linen.normalization import _canonicalize_axes, _compute_stats, _normalize
from flax.training.train_state import TrainState
from flax.traverse_util import flatten_dict, unflatten_dict
from jax._src.distributed import initialize as jax_distributed_initialize
from jax.nn import initializers
from jax.sharding import NamedSharding, PartitionSpec  # noqa
from jax_tqdm import scan_tqdm
from omegaconf import OmegaConf
from safetensors.flax import load_file, save_file
from typing_extensions import NotRequired
from xtils.jitpp import Static, jit

import wandb

logger = logging.getLogger(__name__)

SCRATCH = Path(os.environ["SCRATCH"])

# https://docs.jax.dev/en/latest/persistent_compilation_cache.html#quick-start
jax.config.update("jax_compilation_cache_dir", str(SCRATCH / "jax_cache"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)


PRNGKey = Any
Array = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Axes = Union[int, Sequence[int]]


def get_gpus_per_task(tres_per_task: str) -> int:
    """Returns the number of GPUS per task from the SLURM env variables.

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
        res_type, _, res_count = part.partition(":")
        if res_type == "gres/gpu":
            gpus_per_task = int(res_count.rpartition("=")[-1])
            assert gpus_per_task > 0
            return gpus_per_task
    return 0


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


class BatchRenorm(Module):
    """BatchRenorm Module, implemented based on the Batch Renormalization paper (https://arxiv.org/abs/1702.03275).
    and adapted from Flax's BatchNorm implementation:
    https://github.com/google/flax/blob/ce8a3c74d8d1f4a7d8f14b9fb84b2cc76d7f8dbf/flax/linen/normalization.py#L228
    """

    use_running_average: Optional[bool] = None
    axis: int = -1
    momentum: float = 0.999
    epsilon: float = 0.001
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones
    axis_name: Optional[str] = None
    axis_index_groups: Any = None
    use_fast_variance: bool = True

    @compact
    def __call__(self, x: jax.Array, use_running_average: Optional[bool] = None):
        use_running_average = merge_param(
            "use_running_average", self.use_running_average, use_running_average
        )
        feature_axes = _canonicalize_axes(x.ndim, self.axis)
        reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)
        feature_shape = [x.shape[ax] for ax in feature_axes]
        # todo: Do we have to sync stuff once we make this distributed?
        ra_mean = self.variable(
            "batch_stats", "mean", lambda s: jnp.zeros(s, jnp.float32), feature_shape
        )
        ra_var = self.variable(
            "batch_stats", "var", lambda s: jnp.ones(s, jnp.float32), feature_shape
        )

        r_max = self.variable("batch_stats", "r_max", lambda s: s, 3)
        d_max = self.variable("batch_stats", "d_max", lambda s: s, 5)
        steps = self.variable("batch_stats", "steps", lambda s: s, 0)

        if use_running_average:
            mean, var = ra_mean.value, ra_var.value
            custom_mean = mean
            custom_var = var
        else:
            mean, var = _compute_stats(
                x,
                reduction_axes,
                dtype=self.dtype,
                axis_name=self.axis_name if not self.is_initializing() else None,
                axis_index_groups=self.axis_index_groups,
                use_fast_variance=self.use_fast_variance,
            )
            custom_mean = mean
            custom_var = var
            if not self.is_initializing():
                # The code below is implemented following the Batch Renormalization paper
                r = 1
                d = 0
                assert isinstance(mean, jax.Array)
                assert isinstance(var, jax.Array)
                assert isinstance(custom_var, jax.Array)
                std = jnp.sqrt(var + self.epsilon)
                ra_std = jnp.sqrt(ra_var.value + self.epsilon)
                r = jax.lax.stop_gradient(std / ra_std)
                r = jnp.clip(r, 1 / r_max.value, r_max.value)
                d = jax.lax.stop_gradient((mean - ra_mean.value) / ra_std)
                d = jnp.clip(d, -d_max.value, d_max.value)
                tmp_var = var / (r**2)
                tmp_mean = mean - d * jnp.sqrt(custom_var) / r

                # Warm up batch renorm for 100_000 steps to build up proper running statistics
                warmed_up = jnp.greater_equal(steps.value, 1000).astype(jnp.float32)
                custom_var = warmed_up * tmp_var + (1.0 - warmed_up) * custom_var
                custom_mean = warmed_up * tmp_mean + (1.0 - warmed_up) * custom_mean

                ra_mean.value = (
                    self.momentum * ra_mean.value + (1 - self.momentum) * mean
                )
                ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * var
                steps.value += 1

        return _normalize(
            self,
            x,
            custom_mean,
            custom_var,
            reduction_axes,
            feature_axes,
            self.dtype,
            self.param_dtype,
            self.epsilon,
            self.use_bias,
            self.use_scale,
            self.bias_init,
            self.scale_init,
        )


class ScannedRNN(nn.Module):
    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(
        self, carry: tuple[jax.Array, jax.Array], x: tuple[jax.Array, jax.Array]
    ):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        hidden_size = rnn_state[0].shape[-1]

        init_rnn_state = self.initialize_carry(hidden_size, *resets.shape)
        resets_mask = resets[:, np.newaxis]
        rnn_state = jax.tree.map(
            lambda init, old: jnp.where(resets_mask, init, old),
            init_rnn_state,
            rnn_state,
        )

        new_rnn_state, y = nn.OptimizedLSTMCell(hidden_size)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.OptimizedLSTMCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*batch_size, hidden_size)
        )


class RNNQNetwork(nn.Module):
    action_dim: int
    hidden_size: int = 512
    num_layers: int = 4
    num_rnn_layers: int = 1
    norm_input: bool = False
    norm_type: Literal["layer_norm", "batch_norm"] | None = "layer_norm"
    dueling: bool = False
    add_last_action: bool = False

    @nn.compact
    def __call__(
        self,
        hidden: Sequence[tuple[jax.Array, jax.Array]],
        x: jax.Array,
        done,
        last_action,
        train: bool = False,
    ):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)  # noqa
        elif self.norm_type == "batch_norm":
            normalize = lambda x: BatchRenorm(use_running_average=not train)(x)  # noqa
        else:
            normalize = lambda x: x  # noqa

        if self.norm_input:
            x = BatchRenorm(use_running_average=not train)(x)
        else:
            # dummy normalize input in any case for global compatibility
            x_dummy = BatchRenorm(use_running_average=not train)(x)  # noqa (actually changes something?)

        for _layer_index in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = normalize(x)
            x = nn.relu(x)

        # add last action to the input of the rnn
        if self.add_last_action:
            last_action = jax.nn.one_hot(last_action, self.action_dim)
            x = jnp.concatenate([x, last_action], axis=-1)

        new_hidden: list[tuple[jax.Array, jax.Array]] = []
        # NOTE: shouldn't this be some sort of scan?
        for i in range(self.num_rnn_layers):
            rnn_in = (x, done)
            hidden_aux, x = ScannedRNN()(hidden[i], rnn_in)
            new_hidden.append(hidden_aux)

        q_vals = nn.Dense(self.action_dim)(x)

        return new_hidden, q_vals

    def initialize_carry(self, *batch_size: int):
        return [
            ScannedRNN.initialize_carry(self.hidden_size, *batch_size)
            for _ in range(self.num_rnn_layers)
        ]


class Transition(flax.struct.PyTreeNode):
    last_hs: list[tuple[jax.Array, jax.Array]]
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array
    last_done: jax.Array
    last_action: jax.Array
    q_vals: jax.Array


class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


class _ExplorationState(NamedTuple):
    hs: list[tuple[jax.Array, jax.Array]]
    obs: jax.Array
    done: jax.Array
    action: jax.Array
    env_state: EnvState
    # rng: chex.PRNGKey


class _RunnerState(NamedTuple):
    train_state: CustomTrainState
    # memory_transitions: Transition
    expl_state: _ExplorationState
    test_metrics: dict | None
    rng: chex.PRNGKey


class Results(TypedDict):
    runner_state: _RunnerState
    metrics: dict[str, jax.Array]


class Config(TypedDict):
    SEED: int
    NUM_SEEDS: int
    TOTAL_TIMESTEPS: int
    NUM_STEPS: int
    NUM_ENVS: int
    TOTAL_TIMESTEPS_DECAY: int
    NUM_ENVS: int
    NUM_MINIBATCHES: int
    NUM_EPOCHS: int
    ENV_NAME: str
    EPS_START: float
    EPS_FINISH: float
    EPS_DECAY: int
    EPS_TEST: float
    LR: float
    NORM_TYPE: Literal["layer_norm", "batch_norm"] | None
    NORM_INPUT: NotRequired[bool]
    NORM_LAYERS: NotRequired[int]
    TEST_NUM_ENVS: int
    TEST_INTERVAL: int
    MAX_GRAD_NORM: float
    LAMBDA: float
    GAMMA: float
    REW_SCALE: float
    HIDDEN_SIZE: NotRequired[int]
    NUM_LAYERS: NotRequired[int]
    ENTITY: str
    PROJECT: str
    WANDB_MODE: Literal["disabled", "online", "offline"]
    ALG_NAME: str
    SAVE_PATH: NotRequired[str]
    WANDB_LOG_ALL_SEEDS: NotRequired[bool]
    HYP_TUNE: bool

    # Fields specificic to this file (compared to pqn_gymnax)
    USE_OPTIMISTIC_RESETS: bool
    MEMORY_WINDOW: int
    # Fields that are written-to in `make_train`:
    # todo: move these out of here.
    NUM_UPDATES: int
    NUM_UPDATES_DECAY: int
    OPTIMISTIC_RESET_RATIO: int
    TEST_NUM_STEPS: int


def make_train(config: Config):
    config = copy.deepcopy(config)
    config["NUM_UPDATES"] = (
        int(config["TOTAL_TIMESTEPS"]) // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    config["NUM_UPDATES_DECAY"] = (
        int(config["TOTAL_TIMESTEPS_DECAY"])
        // config["NUM_STEPS"]
        // config["NUM_ENVS"]
    )

    assert (config["NUM_STEPS"] * config["NUM_ENVS"]) % config[
        "NUM_MINIBATCHES"
    ] == 0, "NUM_MINIBATCHES must divide NUM_STEPS*NUM_ENVS"

    basic_env = make_craftax_env_from_name(
        config["ENV_NAME"], auto_reset=not config["USE_OPTIMISTIC_RESETS"]
    )
    env_params = basic_env.default_params
    log_env = LogWrapper(basic_env)
    if config["USE_OPTIMISTIC_RESETS"]:
        env = OptimisticResetVecEnvWrapper(
            log_env,
            num_envs=config["NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
        )
        test_env = OptimisticResetVecEnvWrapper(
            log_env,
            num_envs=config["TEST_NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["TEST_NUM_ENVS"]),
        )
    else:
        env = BatchEnvWrapper(log_env, num_envs=config["NUM_ENVS"])
        test_env = BatchEnvWrapper(log_env, num_envs=config["TEST_NUM_ENVS"])

    return functools.partial(
        train,
        config=FrozenDict(config),
        env=env,
        env_params=env_params,
        test_env=test_env,
    )


def scan_with_progress[
    Carry,
    In: int | jax.Array | tuple[int | jax.Array, ...],
    Out,
](
    fn: Callable[[Carry, In], tuple[Carry, Out]],
    init: Carry,
    xs: In,
    length: int,
    desc: str | None = None,
    **kwargs,
) -> tuple[Carry, Out]:
    if desc:
        kwargs["desc"] = desc
    return jax.lax.scan(
        scan_tqdm(length, **kwargs)(fn),
        init=init,
        xs=xs,
        length=length,
    )


@jit  # called once per node?
def train(
    rng: chex.PRNGKey,
    config: Static[Config],
    env: Static[OptimisticResetVecEnvWrapper],
    env_params: Static[EnvParams],
    test_env: Static[BatchEnvWrapper],
):
    num_envs: int = config["NUM_ENVS"]
    num_updates: int = int(config["NUM_UPDATES"])

    eps_scheduler = optax.linear_schedule(
        config["EPS_START"],
        config["EPS_FINISH"],
        (config["EPS_DECAY"]) * config["NUM_UPDATES_DECAY"],
    )

    lr_scheduler = optax.linear_schedule(
        init_value=config["LR"],
        end_value=1e-20,
        transition_steps=(config["NUM_UPDATES_DECAY"])
        * config["NUM_MINIBATCHES"]
        * config["NUM_EPOCHS"],
    )
    lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

    # INIT NETWORK AND OPTIMIZER
    network = RNNQNetwork(
        action_dim=env.action_space(env_params).n,
        hidden_size=config.get("HIDDEN_SIZE", 128),
        num_layers=config.get("NUM_LAYERS", 2),
        num_rnn_layers=config.get("NUM_RNN_LAYERS", 1),
        norm_type=config["NORM_TYPE"],
        norm_input=config.get("NORM_INPUT", False),
        add_last_action=config.get("ADD_LAST_ACTION", False),
    )
    # NOTE (rng): _rng wasn't used here, renamed to _agent_rng.
    rng, _agent_rng = jax.random.split(rng)
    train_state = create_agent(
        _agent_rng,  # was `rng`.
        env=env,
        env_params=env_params,
        config=config,
        network=network,
        lr=lr,
    )

    # TRAINING LOOP
    rng, _initial_test_metrics_rng = jax.random.split(rng)
    test_metrics = get_test_metrics(
        train_state,
        _initial_test_metrics_rng,
        config=config,
        network=network,
        test_env=test_env,
        env_params=env_params,
    )
    assert test_metrics is None or isinstance(test_metrics, dict)

    # step randomly to have the initial memory window

    # NOTE: `rng` was being overwritten anyway below, using it directly instead.
    # rng, _rng = jax.random.split(rng)

    # todo: Would be nice to be able to start from a checkpoint and maybe skip this step.
    assert config["MEMORY_WINDOW"] == 0
    # NOTE Since the memory window is 0, we can skip all this memory buffer stuff!
    # buffer_length = config["MEMORY_WINDOW"] + config["NUM_STEPS"]
    # (expl_state, rng), memory_transitions = scan_with_progress(
    #     lambda expl_state_and_rng, _step: random_step(
    #         expl_state_and_rng,
    #         _step,
    #         network=network,
    #         train_state=train_state,
    #         env=env,
    #         env_params=env_params,
    #         num_envs=num_envs,
    #         reward_scaling_factor=config["REW_SCALE"],
    #     ),
    #     init=(expl_state, rng),
    #     xs=jnp.arange(buffer_length),
    #     length=buffer_length,
    #     desc="Filling up the initial buffers.",
    # )

    # train
    # NOTE: rng was just unused after this, so using it directly instead of splitting.
    # rng, _rng = jax.random.split(rng)

    rng, _initial_env_state_rng = jax.random.split(rng)
    _initial_obs, _initial_env_state = env.reset(_initial_env_state_rng, env_params)
    expl_state = _ExplorationState(
        hs=network.initialize_carry(num_envs),
        obs=_initial_obs,
        done=jnp.zeros((num_envs), dtype=bool),
        action=jnp.zeros((num_envs), dtype=int),
        env_state=_initial_env_state,
    )

    runner_state = _RunnerState(
        train_state,
        # memory_transitions,
        expl_state,
        test_metrics,
        rng,  # was _rng
    )

    # todo: Would be nice to be able restart from an existing
    # checkpoint by changing this to a jax.lax.fori_loop of some sort instead of a scan!
    current_update = 0
    runner_state, metrics = scan_with_progress(
        lambda runner_state, update: update_step(
            runner_state,
            update,
            config=config,
            network=network,
            env=env,
            env_params=env_params,
            eps_scheduler=eps_scheduler,
            # original_rng=original_rng,
            test_env=test_env,
        ),
        init=runner_state,
        xs=jnp.arange(current_update, num_updates),
        length=num_updates - current_update,
        desc="Training...",
    )

    return Results(runner_state=runner_state, metrics=metrics)


@jit
def random_step(
    carry: tuple[_ExplorationState, chex.PRNGKey],
    _step: jax.Array,
    *,
    network: Static[RNNQNetwork],
    train_state: CustomTrainState,
    env: Static[BatchEnvWrapper],
    env_params: Static[EnvParams],
    num_envs: Static[int],
    reward_scaling_factor: float = 1.0,
):
    expl_state, rng = carry
    hs, last_obs, last_done, last_action, env_state = expl_state
    rng, rng_a, rng_s = jax.random.split(rng, 3)
    _obs = last_obs[np.newaxis]  # (1 (dummy time), num_envs, obs_size)
    _done = last_done[np.newaxis]  # (1 (dummy time), num_envs)
    _last_action = last_action[np.newaxis]  # (1 (dummy time), num_envs)
    new_hs, q_vals = network.apply(
        {
            "params": train_state.params,
            "batch_stats": train_state.batch_stats,
        },
        hs,
        _obs,
        _done,
        _last_action,
        train=False,
    )  # (num_envs, hidden_size), (1, num_envs, num_actions)
    assert isinstance(q_vals, jax.Array)
    q_vals = q_vals.squeeze(axis=0)  # (num_envs, num_actions) remove the time dim
    _rngs = jax.random.split(rng_a, num_envs)
    eps = jnp.full(num_envs, 1.0)  # random actions
    new_action = jax.vmap(eps_greedy_exploration, axis_name="envs")(_rngs, q_vals, eps)
    new_obs, new_env_state, reward, new_done, info = env.step(
        rng_s, env_state, new_action, env_params
    )
    transition = Transition(
        last_hs=hs,
        obs=last_obs,
        action=new_action,
        reward=reward_scaling_factor * reward,
        done=new_done,
        last_done=last_done,
        last_action=last_action,
        q_vals=q_vals,
    )
    new_expl_state = _ExplorationState(
        new_hs,
        new_obs,
        new_done,
        new_action,
        new_env_state,
    )
    return (new_expl_state, rng), transition


@jit
def get_test_metrics(
    train_state: CustomTrainState,
    rng: chex.PRNGKey,
    *,
    config: Static[Config],
    network: Static[RNNQNetwork],
    test_env: Static[BatchEnvWrapper],
    env_params: Static[EnvParams],
):
    test_during_training = config.get("TEST_DURING_TRAINING", False)
    test_num_envs: int = config["TEST_NUM_ENVS"]
    test_num_steps: int = config["TEST_NUM_STEPS"]

    if not test_during_training:
        return None

    rng, _test_rng = jax.random.split(rng)
    init_obs, env_state = test_env.reset(_test_rng, env_params)
    exploration_state = _ExplorationState(
        hs=network.initialize_carry(test_num_envs),  # (n_envs, hs_size)
        obs=init_obs,
        done=jnp.zeros((test_num_envs), dtype=bool),
        action=jnp.zeros((test_num_envs), dtype=int),
        env_state=env_state,
    )

    (_new_exploration_state, _rng), infos = scan_with_progress(
        lambda expl_state_and_rng, _step: _greedy_env_step(
            expl_state_and_rng,
            _step,
            network=network,
            train_state=train_state,
            config=config,
            test_env=test_env,
            test_env_rng=_test_rng,
            env_params=env_params,
        ),
        (exploration_state, _test_rng),
        xs=jnp.arange(test_num_steps),  # None,
        length=test_num_steps,
        desc="Testing...",
    )
    returned_episode = infos["returned_episode"]
    assert isinstance(returned_episode, jax.Array)
    # return mean of done infos
    mean_done_infos = jax.tree.map(
        lambda x: jnp.mean(x, where=returned_episode),
        # lambda x: (x * returned_episode).sum() / returned_episode_sum,
        infos,
    )
    return mean_done_infos


@jit
def _greedy_env_step(
    step_state: tuple[_ExplorationState, chex.PRNGKey],
    _step: jax.Array,
    *,
    network: Static[RNNQNetwork],
    train_state: CustomTrainState,
    config: Static[Config],
    test_env: Static[GymnaxWrapper],
    env_params: Static[EnvParams],
    test_env_rng: chex.PRNGKey,
):
    test_num_envs: int = config["TEST_NUM_ENVS"]
    eps_test: float = config["EPS_TEST"]

    expl_state, rng = step_state
    hs, last_obs, last_done, last_action, env_state = expl_state
    rng, rng_a, rng_s = jax.random.split(rng, 3)
    _obs = last_obs[np.newaxis]  # (1 (dummy time), num_envs, obs_size)
    _done = last_done[np.newaxis]  # (1 (dummy time), num_envs)
    _last_action = last_action[np.newaxis]  # (1 (dummy time), num_envs)
    new_hs, q_vals = network.apply(
        {
            "params": train_state.params,
            "batch_stats": train_state.batch_stats,
        },
        hs,
        _obs,
        _done,
        _last_action,
        train=False,
    )  # (num_envs, hidden_size), (1, num_envs, num_actions)
    assert isinstance(q_vals, jax.Array)
    q_vals = q_vals.squeeze(axis=0)  # (num_envs, num_actions) remove the time dim
    eps = jnp.full(test_num_envs, eps_test)
    new_action = jax.vmap(eps_greedy_exploration)(
        jax.random.split(rng_a, test_num_envs), q_vals, eps
    )
    # TODO: Why does this use `_rng` instead of the rng that is passed in as input?
    new_obs, new_env_state, reward, new_done, info = test_env.step(
        test_env_rng, env_state, new_action, env_params
    )
    new_expl_state = _ExplorationState(
        new_hs, new_obs, new_done, new_action, new_env_state
    )
    return (new_expl_state, rng), info


@jit
def update_step(
    runner_state: _RunnerState,
    step: jax.Array,
    config: Static[Config],
    network: Static[RNNQNetwork],
    env: Static[OptimisticResetVecEnvWrapper],
    env_params: Static[EnvParams],
    eps_scheduler: Static[optax.Schedule],
    test_env: Static[BatchEnvWrapper],
):
    num_steps: int = config["NUM_STEPS"]  # steps per environment in each update
    num_envs: int = config["NUM_ENVS"]
    num_epochs: int = config["NUM_EPOCHS"]  # minibatches per epoch
    num_minibatches: int = config["NUM_MINIBATCHES"]  # minibatches per epoch (also?)
    gamma: float = config["GAMMA"]
    reward_scaling_coefficient: float = config["REW_SCALE"]
    lambda_: float = config["LAMBDA"]

    train_state, exploration_state, test_metrics, rng = runner_state

    # NETWORKS UPDATE
    (train_state, rng), (loss, qvals, infos) = jax.lax.scan(
        lambda train_state_and_rng, epoch: learn_epoch(
            train_state_and_rng,
            epoch,
            network=network,
            num_minibatches=num_minibatches,
            gamma=gamma,
            lambda_=lambda_,
            num_envs=num_envs,
            env=env,
            env_params=env_params,
            eps_scheduler=eps_scheduler,
            reward_scaling_coefficient=reward_scaling_coefficient,
            num_steps_per_update=num_steps,
        ),
        init=(train_state, rng),
        xs=jnp.arange(num_epochs),
        length=num_epochs,
        # desc="Updating networks...",
        # leave=False,
        # position=1,
    )
    # update timesteps count
    train_state = dataclasses.replace(
        train_state,
        n_updates=train_state.n_updates + 1,
        timesteps=train_state.timesteps + num_steps * num_envs,
    )

    assert isinstance(loss, jax.Array)
    assert isinstance(qvals, jax.Array)
    metrics = {
        "env_step": train_state.timesteps,
        "update_steps": train_state.n_updates,
        "grad_steps": train_state.grad_steps,
        "td_loss": loss.mean(),
        "qvals": qvals.mean(),
    }
    returned_episode = infos["returned_episode"]
    assert isinstance(returned_episode, jax.Array)

    done_infos = jax.tree.map(
        lambda x: jnp.mean(x, where=returned_episode),
        # lambda x: (x * returned_episode).sum() / infos["returned_episode"].sum(),
        infos,
    )
    metrics.update(done_infos)

    # TODO: bad. Should be happening in a scan or something similar, no?
    # if config.get("TEST_DURING_TRAINING", False):
    #     rng, _rng = jax.random.split(rng)
    #     # doesn't this compute the test metrics at every step in any case (due to jit?)
    #     test_metrics = jax.lax.cond(
    #         train_state.n_updates % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"])
    #         == 0,
    #         lambda _: get_test_metrics(
    #             train_state,
    #             _rng,
    #             config=config,
    #             network=network,
    #             test_env=test_env,
    #             env_params=env_params,
    #         ),
    #         lambda _: test_metrics,
    #         operand=None,
    #     )
    #     metrics.update({f"test_{k}": v for k, v in test_metrics.items()})

    # remove achievement metrics if not logging them
    if not config.get("LOG_ACHIEVEMENTS", False):
        metrics = {k: v for k, v in metrics.items() if "achievement" not in k.lower()}

    # report on wandb if required
    # if config["WANDB_MODE"] != "disabled" and jax.process_index() == 0:

    #     def callback(metrics):
    #         # if config.get("WANDB_LOG_ALL_SEEDS", False):
    #         #     metrics.update(
    #         #         {f"rng{int(original_rng)}/{k}": v for k, v in metrics.items()}
    #         #     )
    #         wandb.log(metrics, step=metrics["update_steps"])

    #     jax.debug.callback(callback, metrics, ordered=False)

    runner_state = _RunnerState(
        train_state,
        # transition_buffer,
        exploration_state,
        test_metrics,
        rng,
    )

    return runner_state, metrics


def collect_transitions_and_update_buffer(
    exploration_state: _ExplorationState,
    rng: chex.PRNGKey,
    num_steps: int,
    transition_buffer: Transition,
    network: Static[RNNQNetwork],
    train_state: CustomTrainState,
    eps_scheduler: Static[optax.Schedule],
    env: Static[BatchEnvWrapper],
    env_params: Static[EnvParams],
    num_envs: Static[int],
    reward_scaling_coefficient: float = 1.0,
):
    (exploration_state, rng), (_transitions, infos) = jax.lax.scan(
        lambda expl_state_and_rng, _step: step_env(
            expl_state_and_rng,
            _step,
            network=network,
            train_state=train_state,
            eps_scheduler=eps_scheduler,
            env=env,
            env_params=env_params,
            num_envs=num_envs,
            reward_scaling_coefficient=reward_scaling_coefficient,
        ),
        init=(exploration_state, rng),
        xs=jnp.arange(num_steps),  # None
        length=num_steps,
    )
    transition_buffer = jax.tree.map(
        lambda x, y: jnp.concatenate([x[num_steps:], y], axis=0),
        transition_buffer,
        _transitions,
    )
    return exploration_state, rng, transition_buffer, infos


@jit
def learn_epoch(
    carry: tuple[CustomTrainState, chex.PRNGKey],
    _epoch: jax.Array,
    *,
    num_minibatches: Static[int],
    network: Static[RNNQNetwork],
    gamma: float,
    lambda_: float,
    num_envs: Static[int],
    env: Static[OptimisticResetVecEnvWrapper],
    env_params: Static[EnvParams],
    eps_scheduler: Static[optax.Schedule],
    reward_scaling_coefficient: float,
    num_steps_per_update: Static[int],
):
    train_state, rng = carry

    rng, env_rng, shuffle_transitions_key = jax.random.split(rng, 3)

    _initial_obs, _initial_env_state = env.reset(env_rng, env_params)
    expl_state = _ExplorationState(
        hs=network.initialize_carry(num_envs),
        obs=_initial_obs,
        done=jnp.zeros((num_envs), dtype=bool),
        action=jnp.zeros((num_envs), dtype=int),
        env_state=_initial_env_state,
    )
    (expl_state, rng), (transitions, infos) = jax.lax.scan(
        lambda expl_state_and_rng, _step: step_env(
            expl_state_and_rng,
            _step,
            network=network,
            train_state=train_state,
            eps_scheduler=eps_scheduler,
            env=env,
            env_params=env_params,
            num_envs=num_envs,
            reward_scaling_coefficient=reward_scaling_coefficient,
        ),
        init=(expl_state, rng),
        xs=jnp.arange(num_steps_per_update),  # None
        length=num_steps_per_update,
    )

    # Shuffle and reshape into (minibatches, num_steps, batch_size/num_minbatches, ...)
    transitions = jax.tree_util.tree_map(
        lambda x: preprocess_transition(
            x, shuffle_transitions_key, num_minibatches=num_minibatches
        ),
        transitions,
    )
    train_state, (loss, qvals) = jax.lax.scan(
        lambda train_state, minibatch: update_network(
            train_state,
            minibatch,
            network=network,
            gamma=gamma,
            lambda_=lambda_,
        ),
        init=train_state,
        xs=transitions,
    )

    return (train_state, rng), (loss, qvals, infos)


@jit
def update_network(
    carry: CustomTrainState,
    minibatch: Transition,
    *,
    network: Static[RNNQNetwork],
    gamma: float,
    lambda_: float,
):
    # minibatch shape: num_steps, batch_size, ...
    # with batch_size = num_envs/num_minibatches

    # NOTE: `rng` isn't used here (at all!), removed.
    # train_state, rng = carry
    train_state = carry

    # hs of oldest step (batch_size, hidden_size)
    hs = jax.tree.map(lambda x: x[0], minibatch.last_hs)
    agent_in = (
        minibatch.obs,
        minibatch.last_done,
        minibatch.last_action,
    )

    (loss, (updates, qvals)), grads = jax.value_and_grad(
        lambda params: loss_fn(
            params,
            network=network,
            train_state=train_state,
            hs=hs,
            agent_in=agent_in,
            minibatch=minibatch,
            gamma=gamma,
            lambda_=lambda_,
        ),
        has_aux=True,
    )(train_state.params)

    # TODO: Rearrange this whole thing to be simpler, like in Stoix repo.
    # grads = jax.lax.pmean(grads, axis_name="devices")

    train_state = train_state.apply_gradients(grads=grads)
    train_state = train_state.replace(
        grad_steps=train_state.grad_steps + 1,
        batch_stats=updates["batch_stats"],
    )
    return train_state, (loss, qvals)


@jit
def loss_fn(
    params,
    network: Static[RNNQNetwork],
    train_state: CustomTrainState,
    hs: jax.Array,
    agent_in: tuple[jax.Array, ...],
    minibatch: Transition,
    gamma: float,
    lambda_: float,
):
    (_, q_vals), updates = partial(network.apply, train=True, mutable=["batch_stats"])(
        {"params": params, "batch_stats": train_state.batch_stats},
        hs,
        *agent_in,
    )  # (num_steps, batch_size, num_actions)

    # lambda returns are computed using NUM_STEPS as the horizon, and optimizing from t=0 to NUM_STEPS-1
    target_q_vals = jax.lax.stop_gradient(q_vals)
    last_q = target_q_vals[-1].max(axis=-1)
    target = _compute_targets(
        last_q,  # q_vals at t=NUM_STEPS-1
        target_q_vals[:-1],
        minibatch.reward[:-1],
        minibatch.done[:-1],
        gamma=gamma,
        lambda_=lambda_,
    ).reshape(-1)  # (num_steps-1*batch_size,)

    chosen_action_qvals = jnp.take_along_axis(
        q_vals,
        jnp.expand_dims(minibatch.action, axis=-1),
        axis=-1,
    ).squeeze(axis=-1)  # (num_steps, num_agents, batch_size,)
    chosen_action_qvals = chosen_action_qvals[:-1].reshape(
        -1
    )  # (num_steps-1*batch_size,)

    loss = 0.5 * jnp.square(chosen_action_qvals - target).mean()

    return loss, (updates, chosen_action_qvals)


@jit
def _compute_targets(
    last_q: jax.Array,
    q_vals: jax.Array,
    reward: jax.Array,
    done: jax.Array,
    gamma: float,
    lambda_: float,
):
    lambda_returns = reward[-1] + gamma * (1 - done[-1]) * last_q
    last_q = jnp.max(q_vals[-1], axis=-1)
    _, targets = jax.lax.scan(
        lambda _lambda_returns_and_last_q, _rew_qval_done: _get_target(
            _lambda_returns_and_last_q, _rew_qval_done, gamma=gamma, lambda_=lambda_
        ),
        (lambda_returns, last_q),
        jax.tree.map(lambda x: x[:-1], (reward, q_vals, done)),
        reverse=True,
    )
    targets = jnp.concatenate([targets, lambda_returns[np.newaxis]])
    return targets


@jit
def _get_target(
    lambda_returns_and_next_q: tuple[jax.Array, jax.Array],
    rew_q_done: tuple[jax.Array, jax.Array, jax.Array],
    *,
    gamma: float,
    lambda_: float,
):
    reward, q, done = rew_q_done
    lambda_returns, next_q = lambda_returns_and_next_q
    target_bootstrap = reward + gamma * (1 - done) * next_q
    delta = lambda_returns - next_q
    lambda_returns = target_bootstrap + gamma * lambda_ * delta
    lambda_returns = (1 - done) * lambda_returns + done * reward
    next_q = jnp.max(q, axis=-1)
    return (lambda_returns, next_q), lambda_returns


@jit
def preprocess_transition(x: jax.Array, rng: jax.Array, num_minibatches: Static[int]):
    # x: (num_steps, num_envs, ...)
    x = jax.random.permutation(rng, x, axis=1)  # shuffle the transitions
    x = x.reshape(
        x.shape[0], num_minibatches, -1, *x.shape[2:]
    )  # num_steps, minibatches, batch_size/num_minbatches,
    x = jnp.swapaxes(
        x, 0, 1
    )  # (minibatches, num_steps, batch_size/num_minbatches, ...)
    return x


@jit
def step_env(
    carry: tuple[_ExplorationState, chex.PRNGKey],
    _step: jax.Array,
    *,
    network: Static[RNNQNetwork],
    train_state: CustomTrainState,
    eps_scheduler: Static[optax.Schedule],
    env: Static[BatchEnvWrapper | OptimisticResetVecEnvWrapper],
    env_params: Static[EnvParams],
    num_envs: Static[int],
    reward_scaling_coefficient: float = 1.0,
):
    _exploration_state, rng = carry
    hs, last_obs, last_done, last_action, env_state = _exploration_state
    rng, rng_a, rng_s = jax.random.split(rng, 3)

    _obs = last_obs[np.newaxis]  # (1 (dummy time), num_envs, obs_size)
    _done = last_done[np.newaxis]  # (1 (dummy time), num_envs)
    _last_action = last_action[np.newaxis]  # (1 (dummy time), num_envs)

    new_hs, q_vals = network.apply(
        {
            "params": train_state.params,
            "batch_stats": train_state.batch_stats,
        },
        hs,
        _obs,
        _done,
        _last_action,
        train=False,
    )  # (num_envs, hidden_size), (1, num_envs, num_actions)
    assert isinstance(q_vals, jax.Array)
    q_vals = q_vals.squeeze(axis=0)  # (num_envs, num_actions) remove the time dim

    _rngs = jax.random.split(rng_a, num_envs)
    eps = jnp.full(num_envs, eps_scheduler(train_state.n_updates))
    new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)

    new_obs, new_env_state, reward, new_done, info = env.step(
        rng_s, env_state, new_action, env_params
    )

    transition = Transition(
        last_hs=hs,
        obs=last_obs,
        action=new_action,
        reward=reward_scaling_coefficient * reward,
        done=new_done,
        last_done=last_done,
        last_action=last_action,
        q_vals=q_vals,
    )
    new_expl_state = _ExplorationState(
        new_hs, new_obs, new_done, new_action, new_env_state
    )
    return (new_expl_state, rng), (transition, info)


@jit
def create_agent(
    rng: chex.PRNGKey,
    env: Static[OptimisticResetVecEnvWrapper],
    env_params: Static[EnvParams],
    config: Static[Config],
    network: Static[RNNQNetwork],
    lr: Static[float | jax.Array | optax.Schedule],
):
    init_x = (
        jnp.zeros(
            (1, 1, *env.observation_space(env_params).shape)
        ),  # (time_step, batch_size, obs_size)
        jnp.zeros((1, 1)),  # (time_step, batch size)
        jnp.zeros((1, 1)),  # (time_step, batch size)
    )  # (obs, dones, last_actions)
    init_hs = network.initialize_carry(1)  # (batch_size, hidden_dim)
    network_variables = network.init(rng, init_hs, *init_x, train=False)
    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.radam(learning_rate=lr),
    )

    train_state = CustomTrainState.create(
        apply_fn=network.apply,
        params=network_variables["params"],
        batch_stats=network_variables["batch_stats"],
        tx=tx,
    )
    return train_state


# epsilon-greedy exploration
def eps_greedy_exploration(rng: chex.PRNGKey, q_vals: jax.Array, eps: jax.Array):
    rng_a, rng_e = jax.random.split(
        rng
    )  # a key for sampling random actions and one for picking
    greedy_actions = jnp.argmax(q_vals, axis=-1)
    chosed_actions = jnp.where(
        jax.random.uniform(rng_e, greedy_actions.shape)
        < eps,  # pick the actions that should be random
        jax.random.randint(
            rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
        ),  # sample random actions,
        greedy_actions,
    )
    return chosed_actions


# Including this here to make this compatible with jaxmarl 0.0.4
# (seems like those functions were removed or moved to a different place?)


def save_params(params: dict, filename: str | os.PathLike) -> None:
    flattened_dict = flatten_dict(params, sep=",")
    save_file(flattened_dict, filename)  # type: ignore


def load_params(filename: str | os.PathLike) -> dict:
    flattened_dict = load_file(filename)
    return unflatten_dict(flattened_dict, sep=",")


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


@hydra.main(version_base=None, config_path="./config", config_name="pqn_rnn_craftax")
def main(_config):
    distributed_env = SlurmDistributedEnv()
    setup_logging(distributed_env.local_rank, distributed_env.num_nodes, 2)

    _config = OmegaConf.to_container(_config)
    assert isinstance(_config, dict)
    if distributed_env.global_rank == 0:
        print("Config:\n", OmegaConf.to_yaml(_config))

    # todo: This pattern is bad, why is there even an `alg` config group? Makes no sense.
    config: Config = {**_config, **_config["alg"]}  # type: ignore

    alg_name = config.get("ALG_NAME", "pqn_rnn")
    env_name = config["ENV_NAME"]
    # task_gpus = list(
    #     range(
    #         distributed_env.local_rank * distributed_env.gpus_per_task,
    #         (distributed_env.local_rank + 1) * distributed_env.gpus_per_task,
    #     )
    # )
    task_gpus = list(range(distributed_env.gpus_per_task))
    # todo: adjust if we want to do one task per gpu.
    jax_distributed_initialize(local_device_ids=task_gpus)

    _run = wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
        name=config.get("NAME", f"{config['ALG_NAME']}_{config['ENV_NAME']}"),
        id=f"{distributed_env.job_id}_{distributed_env.global_rank}",
        group=str(distributed_env.job_id),
        config=dict(config),
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])

    # jax.debug.visualize_array_sharding(rngs)

    # TODO: Look into `flax.jax_utils.replicate` and how it is used in the Stoix repo
    # https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/q_learning/ff_qr_dqn.py

    # TODO: Add profiling hooks following https://docs.jax.dev/en/latest/profiling.html
    train_fn = make_train(config)
    logger.info("Starting to jit the training function.")
    _start = time.time()
    # TODO: Disabling multiple seeds for now, to make it simpler to learn how to distribute
    # this across devices.
    # num_seeds = config["NUM_SEEDS"]
    # rngs = jax.random.split(rng, num_seeds)
    # train_fn = jax.jit(jax.vmap(train_fn, axis_name="seeds")).lower(rngs).compile()

    rngs = rng

    # todo: makes no sense, each function gets the same arguments, so produces the same data!
    # train_fn = jax.pmap(train_fn, axis_name="devices", devices=jax.devices())
    # rngs = jax.random.split(rng, len(jax.devices()))

    train_fn = jax.jit(train_fn).lower(rngs).compile()
    logger.info(f"Took {time.time() - _start} seconds to jit.")

    _start = time.time()
    outs: Results = jax.block_until_ready(train_fn(rngs))
    logger.info(f"Took {time.time() - _start} seconds to complete.")
    print(jax.tree.map(jnp.shape, outs["metrics"]))

    # mean_metrics = jax.lax.pmean(outs["metrics"], ("x", "y"))
    # print(jax.tree.map(jnp.shape, mean_metrics))

    if (save_path := config.get("SAVE_PATH")) is not None:
        model_state = outs["runner_state"][0]
        save_dir = os.path.join(save_path, env_name)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(
            config,
            os.path.join(
                save_dir, f"{alg_name}_{env_name}_seed{config['SEED']}_config.yaml"
            ),
        )

        for i, rng in enumerate(rngs):
            params = jax.tree.map(lambda x: x[i], model_state.params)
            save_path = os.path.join(
                save_dir,
                f"{alg_name}_{env_name}_seed{config['SEED']}_vmap{i}.safetensors",
            )
            save_params(params, save_path)

    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
