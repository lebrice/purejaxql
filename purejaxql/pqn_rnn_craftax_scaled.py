"""
This script uses BatchRenorm for more effective batch normalization in long training runs.

```
JAX_TRACEBACK_FILTERING=off srun --pty --gpus-per-task=4 --ntasks-per-node=1 --nodes=1 --cpus-per-task=48 --mem=0 --partition=gpubase_bynode_b3 \
    uv run python purejaxql/pqn_rnn_craftax_scaled.py +alg=pqn_rnn_craftax alg.TOTAL_TIMESTEPS=100 alg.TOTAL_TIMESTEPS_DECAY=100 NUM_SEEDS=4
```
"""

import dataclasses
import functools
import os
import time
import copy
import jax
import jax.experimental
import jax.experimental.mesh_utils
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from jax._src.distributed import initialize as jax_distributed_initialize
import flax.struct
import numpy as np
from functools import partial
from typing import Any, Generic, NamedTuple, TypedDict, Literal
from typing_extensions import NotRequired
from craftax.craftax.craftax_state import EnvParams, EnvState
import chex
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
import hydra
from omegaconf import OmegaConf
import wandb
from xtils.jitpp import Static, jit
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.sharding import PartitionSpec, NamedSharding  # noqa
from safetensors.flax import load_file, save_file

from craftax.craftax_env import make_craftax_env_from_name
from craftax_wrappers import (
    GymnaxWrapper,
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
)

from flax.linen.normalization import _compute_stats, _normalize, _canonicalize_axes
from typing import Callable, Optional, Sequence, Tuple, Union
from flax.linen.module import Module, compact, merge_param
from jax.nn import initializers


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
    def __call__(self, carry, x: tuple[Any, jax.Array]):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        hidden_size = rnn_state[0].shape[-1]

        init_rnn_state = self.initialize_carry(hidden_size, *resets.shape)
        rnn_state = jax.tree.map(
            lambda init, old: jnp.where(resets[:, np.newaxis], init, old),
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
        self, hidden: jax.Array, x: jax.Array, done, last_action, train: bool = False
    ):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: BatchRenorm(use_running_average=not train)(x)
        else:
            normalize = lambda x: x

        if self.norm_input:
            x = BatchRenorm(use_running_average=not train)(x)
        else:
            # dummy normalize input in any case for global compatibility
            x_dummy = BatchRenorm(use_running_average=not train)(x)

        for l in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = normalize(x)
            x = nn.relu(x)

        # add last action to the input of the rnn
        if self.add_last_action:
            last_action = jax.nn.one_hot(last_action, self.action_dim)
            x = jnp.concatenate([x, last_action], axis=-1)

        new_hidden = []
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
    memory_transitions: Transition
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
    REW_SCALE: NotRequired[float]
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
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    config["NUM_UPDATES_DECAY"] = (
        config["TOTAL_TIMESTEPS_DECAY"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    assert (config["NUM_STEPS"] * config["NUM_ENVS"]) % config[
        "NUM_MINIBATCHES"
    ] == 0, "NUM_MINIBATCHES must divide NUM_STEPS*NUM_ENVS"

    basic_env = make_craftax_env_from_name(
        config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"]
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


@jit
def train(
    rng: chex.PRNGKey,
    config: Static[Config],
    env: Static[BatchEnvWrapper],
    env_params: Static[EnvParams],
    test_env: Static[BatchEnvWrapper],
):
    original_rng = rng[0]
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
    rng, _rng = jax.random.split(rng)
    train_state = create_agent(
        rng,
        env=env,
        env_params=env_params,
        config=config,
        network=network,
        lr=lr,
    )

    # TRAINING LOOP
    rng, _rng = jax.random.split(rng)
    test_metrics = get_test_metrics(
        train_state,
        _rng,
        config=config,
        network=network,
        test_env=test_env,
        env_params=env_params,
    )
    assert test_metrics is None or isinstance(test_metrics, dict)

    rng, _rng = jax.random.split(rng)
    obs, env_state = env.reset(_rng, env_params)
    init_dones = jnp.zeros((config["NUM_ENVS"]), dtype=bool)
    init_action = jnp.zeros((config["NUM_ENVS"]), dtype=int)
    init_hs = network.initialize_carry(config["NUM_ENVS"])
    expl_state = _ExplorationState(init_hs, obs, init_dones, init_action, env_state)

    # step randomly to have the initial memory window

    rng, _rng = jax.random.split(rng)

    (expl_state, rng), memory_transitions = jax.lax.scan(
        lambda expl_state_and_rng, _step: random_step(
            expl_state_and_rng,
            _step,
            network=network,
            train_state=train_state,
            config=config,
            env=env,
            env_params=env_params,
        ),
        (expl_state, _rng),
        xs=jnp.arange(config["MEMORY_WINDOW"] + config["NUM_STEPS"]),
        length=config["MEMORY_WINDOW"] + config["NUM_STEPS"],
    )
    # expl_state = tuple(expl_state)

    # train
    rng, _rng = jax.random.split(rng)
    runner_state = _RunnerState(
        train_state, memory_transitions, expl_state, test_metrics, _rng
    )
    runner_state, metrics = jax.lax.scan(
        lambda runner_state, _step: update_step(
            runner_state,
            _step,
            config=config,
            network=network,
            env=env,
            env_params=env_params,
            eps_scheduler=eps_scheduler,
            original_rng=original_rng,
            test_env=test_env,
        ),
        init=runner_state,
        xs=jnp.arange(config["NUM_UPDATES"]),  # None
        length=config["NUM_UPDATES"],
    )

    return Results(runner_state=runner_state, metrics=metrics)


@jit
def random_step(
    carry: tuple[_ExplorationState, chex.PRNGKey],
    _step: jax.Array,
    *,
    network: Static[RNNQNetwork],
    train_state: CustomTrainState,
    config: Static[Config],
    env: Static[BatchEnvWrapper],
    env_params: Static[EnvParams],
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
    _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
    eps = jnp.full(config["NUM_ENVS"], 1.0)  # random actions
    new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)
    new_obs, new_env_state, reward, new_done, info = env.step(
        rng_s, env_state, new_action, env_params
    )
    transition = Transition(
        last_hs=hs,
        obs=last_obs,
        action=new_action,
        reward=config.get("REW_SCALE", 1) * reward,
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
    test_env: Static[GymnaxWrapper],
    env_params: Static[EnvParams],
):
    if not config.get("TEST_DURING_TRAINING", False):
        return None

    rng, _rng = jax.random.split(rng)
    init_obs, env_state = test_env.reset(_rng, env_params)
    init_done = jnp.zeros((config["TEST_NUM_ENVS"]), dtype=bool)
    init_action = jnp.zeros((config["TEST_NUM_ENVS"]), dtype=int)
    init_hs = network.initialize_carry(config["TEST_NUM_ENVS"])  # (n_envs, hs_size)
    exploration_state = _ExplorationState(
        init_hs, init_obs, init_done, init_action, env_state
    )

    (_new_exploration_state, _rng), infos = jax.lax.scan(
        lambda expl_state_and_rng, _step: _greedy_env_step(
            expl_state_and_rng,
            _step,
            network=network,
            train_state=train_state,
            config=config,
            test_env=test_env,
            _rng=_rng,
            env_params=env_params,
        ),
        (exploration_state, _rng),
        xs=jnp.arange(config["TEST_NUM_STEPS"]),  # None,
        length=config["TEST_NUM_STEPS"],
    )
    # return mean of done infos
    done_infos = jax.tree.map(
        lambda x: (x * infos["returned_episode"]).sum()
        / infos["returned_episode"].sum(),
        infos,
    )
    return done_infos


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
    _rng: chex.PRNGKey,
):
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
    eps = jnp.full(config["TEST_NUM_ENVS"], config["EPS_TEST"])
    new_action = jax.vmap(eps_greedy_exploration)(
        jax.random.split(rng_a, config["TEST_NUM_ENVS"]), q_vals, eps
    )
    # TODO: Why does this use `_rng` instead of the rng that is passed in as input?
    new_obs, new_env_state, reward, new_done, info = test_env.step(
        _rng, env_state, new_action, env_params
    )
    new_expl_state = _ExplorationState(
        new_hs, new_obs, new_done, new_action, new_env_state
    )
    return (new_expl_state, rng), info


@jit
def update_step(
    runner_state: _RunnerState,
    _step: jax.Array,
    config: Static[Config],
    network: Static[RNNQNetwork],
    env: Static[BatchEnvWrapper],
    env_params: Static[EnvParams],
    eps_scheduler: Static[optax.Schedule],
    original_rng: chex.PRNGKey,
    test_env: Static[GymnaxWrapper],
):
    train_state, memory_transitions, expl_state, test_metrics, rng = runner_state

    # SAMPLE PHASE
    # step the env
    rng, _rng = jax.random.split(rng)

    # _step_env = functools.partial(
    #     step_env,

    # )

    (expl_state, rng), (transitions, infos) = jax.lax.scan(
        lambda expl_state_and_rng, _step: step_env(
            expl_state_and_rng,
            _step,
            network=network,
            train_state=train_state,
            config=config,
            eps_scheduler=eps_scheduler,
            env=env,
            env_params=env_params,
        ),
        init=(expl_state, _rng),
        xs=jnp.arange(config["NUM_STEPS"]),  # None
        length=config["NUM_STEPS"],
    )
    # expl_state = tuple(expl_state)

    train_state = dataclasses.replace(
        train_state,
        timesteps=train_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"],
    )  # update timesteps count

    # insert the transitions into the memory
    memory_transitions = jax.tree.map(
        lambda x, y: jnp.concatenate([x[config["NUM_STEPS"] :], y], axis=0),
        memory_transitions,
        transitions,
    )

    # NETWORKS UPDATE
    rng, _rng = jax.random.split(rng)
    (train_state, rng), (loss, qvals) = jax.lax.scan(
        lambda train_state_and_rng, _epoch: learn_epoch(
            train_state_and_rng,
            _epoch,
            config=config,
            memory_transitions=memory_transitions,
            network=network,
        ),
        init=(train_state, rng),
        xs=jnp.arange(config["NUM_EPOCHS"]),
        length=config["NUM_EPOCHS"],
    )
    assert isinstance(loss, jax.Array)
    assert isinstance(qvals, jax.Array)

    train_state = dataclasses.replace(train_state, n_updates=train_state.n_updates + 1)
    metrics = {
        "env_step": train_state.timesteps,
        "update_steps": train_state.n_updates,
        "grad_steps": train_state.grad_steps,
        "td_loss": loss.mean(),
        "qvals": qvals.mean(),
    }
    done_infos = jax.tree.map(
        lambda x: (x * infos["returned_episode"]).sum()
        / infos["returned_episode"].sum(),
        infos,
    )
    metrics.update(done_infos)

    if config.get("TEST_DURING_TRAINING", False):
        rng, _rng = jax.random.split(rng)
        test_metrics = jax.lax.cond(
            train_state.n_updates % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"])
            == 0,
            lambda _: get_test_metrics(
                train_state,
                _rng,
                config=config,
                network=network,
                test_env=test_env,
                env_params=env_params,
            ),
            lambda _: test_metrics,
            operand=None,
        )
        metrics.update({f"test_{k}": v for k, v in test_metrics.items()})

    # remove achievement metrics if not logging them
    if not config.get("LOG_ACHIEVEMENTS", False):
        metrics = {k: v for k, v in metrics.items() if "achievement" not in k.lower()}

    # report on wandb if required
    if config["WANDB_MODE"] != "disabled":

        def callback(metrics, original_rng):
            if config.get("WANDB_LOG_ALL_SEEDS", False):
                metrics.update(
                    {f"rng{int(original_rng)}/{k}": v for k, v in metrics.items()}
                )
            wandb.log(metrics, step=metrics["update_steps"])

        jax.debug.callback(callback, metrics, original_rng)

    runner_state = _RunnerState(
        train_state,
        memory_transitions,
        expl_state,
        test_metrics,
        rng,
    )

    return runner_state, metrics


@jit
def learn_epoch(
    carry: tuple[CustomTrainState, chex.PRNGKey],
    _epoch: jax.Array,
    *,
    memory_transitions: Transition,
    config: Static[Config],
    network: Static[RNNQNetwork],
):
    train_state, rng = carry

    rng, _rng = jax.random.split(rng)
    minibatches = jax.tree_util.tree_map(
        lambda x: preprocess_transition(x, _rng, config=config),
        memory_transitions,
    )  # num_minibatches, num_steps+memory_window, batch_size/num_minbatches, ...

    rng, _rng = jax.random.split(rng)
    (train_state, rng), (loss, qvals) = jax.lax.scan(
        lambda train_state_and_rng, minibatch: _learn_phase(
            train_state_and_rng, minibatch, network=network, config=config
        ),
        init=(train_state, rng),
        xs=minibatches,
    )

    return (train_state, rng), (loss, qvals)


@jit
def _learn_phase(
    carry: tuple[CustomTrainState, chex.PRNGKey],
    minibatch: Transition,
    *,
    network: Static[RNNQNetwork],
    config: Static[Config],
):
    # minibatch shape: num_steps, batch_size, ...
    # with batch_size = num_envs/num_minibatches

    train_state, rng = carry
    hs = jax.tree.map(
        lambda x: x[0], minibatch.last_hs
    )  # hs of oldest step (batch_size, hidden_size)
    agent_in = (
        minibatch.obs,
        minibatch.last_done,
        minibatch.last_action,
    )

    (loss, (updates, qvals)), grads = jax.value_and_grad(
        lambda params: _loss_fn(
            params,
            network=network,
            train_state=train_state,
            hs=hs,
            agent_in=agent_in,
            minibatch=minibatch,
            config=config,
        ),
        has_aux=True,
    )(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)
    train_state = train_state.replace(
        grad_steps=train_state.grad_steps + 1,
        batch_stats=updates["batch_stats"],
    )
    return (train_state, rng), (loss, qvals)


@jit
def _loss_fn(
    params,
    network: Static[RNNQNetwork],
    train_state: CustomTrainState,
    hs: jax.Array,
    agent_in: tuple[jax.Array, ...],
    minibatch: Transition,
    config: Static[Config],
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
        config=config,
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
    config: Static[Config],
):
    lambda_returns = reward[-1] + config["GAMMA"] * (1 - done[-1]) * last_q
    last_q = jnp.max(q_vals[-1], axis=-1)
    _, targets = jax.lax.scan(
        lambda _lambda_returns_and_last_q, _rew_qval_done: _get_target(
            _lambda_returns_and_last_q, _rew_qval_done, config=config
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
    config: Static[Config],
):
    reward, q, done = rew_q_done
    lambda_returns, next_q = lambda_returns_and_next_q
    target_bootstrap = reward + config["GAMMA"] * (1 - done) * next_q
    delta = lambda_returns - next_q
    lambda_returns = target_bootstrap + config["GAMMA"] * config["LAMBDA"] * delta
    lambda_returns = (1 - done) * lambda_returns + done * reward
    next_q = jnp.max(q, axis=-1)
    return (lambda_returns, next_q), lambda_returns


@jit
def preprocess_transition(x: jax.Array, rng: jax.Array, config: Static[Config]):
    # x: (num_steps, num_envs, ...)
    x = jax.random.permutation(rng, x, axis=1)  # shuffle the transitions
    x = x.reshape(
        x.shape[0], config["NUM_MINIBATCHES"], -1, *x.shape[2:]
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
    config: Static[Config],
    eps_scheduler: Static[optax.Schedule],
    env: Static[BatchEnvWrapper],
    env_params: Static[EnvParams],
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

    _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
    eps = jnp.full(config["NUM_ENVS"], eps_scheduler(train_state.n_updates))
    new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)

    new_obs, new_env_state, reward, new_done, info = env.step(
        rng_s, env_state, new_action, env_params
    )

    transition = Transition(
        last_hs=hs,
        obs=last_obs,
        action=new_action,
        reward=config.get("REW_SCALE", 1) * reward,
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
    env: Static[BatchEnvWrapper],
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


def single_run(_config: dict):
    distributed_env = SlurmDistributedEnv()
    config: Config = {**_config, **_config["alg"]}  # type: ignore

    alg_name = config.get("ALG_NAME", "pqn_rnn")
    env_name = config["ENV_NAME"]

    # jax_distributed_initialize(
    #     local_device_ids=list(range(distributed_env.gpus_per_task))
    # )

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
        name=config.get("NAME", f"{config['ALG_NAME']}_{config['ENV_NAME']}"),
        id=f"{os.environ['SLURM_JOB_ID']}_{os.environ['SLURM_PROCID']}",  # good idea or not?
        group=os.environ["SLURM_JOB_ID"],
        config=dict(config),
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])

    t0 = time.time()
    num_seeds = config["NUM_SEEDS"]
    # assert num_seeds % 2 == 0, "Debugging distributed stuff for now."
    rngs = jax.random.split(rng, num_seeds)
    # gpus_on_node = distributed_env.gpus_per_task
    # mesh = jax.make_mesh(
    #     (gpus_on_node, num_seeds // gpus_on_node), PartitionSpec("x", "y")
    # )
    # rngs = jax.device_put(rngs, NamedSharding(mesh, PartitionSpec("x", "y")))
    # jax.debug.visualize_array_sharding(rngs)

    train_fn = make_train(config)
    outs = jax.block_until_ready(jax.jit(jax.vmap(train_fn))(rngs))
    print(f"Took {time.time() - t0} seconds to complete.")

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


# Including this here to make this compatible with jaxmarl 0.0.4
# (seems like those functions were removed or moved to a different place?)


def save_params(params: dict, filename: str | os.PathLike) -> None:
    flattened_dict = flatten_dict(params, sep=",")
    save_file(flattened_dict, filename)  # type: ignore


def load_params(filename: str | os.PathLike) -> dict:
    flattened_dict = load_file(filename)
    return unflatten_dict(flattened_dict, sep=",")


def tune(_default_config: dict):
    """Hyperparameter sweep with wandb."""

    default_config: Config = {**_default_config, **_default_config["alg"]}  # type: ignore
    alg_name = default_config.get("ALG_NAME", "pqn")
    env_name = default_config["ENV_NAME"]

    sweep_config = {
        "name": f"{alg_name}_{env_name}",
        "method": "bayes",
        "metric": {
            "name": "test_returned_episode_returns",
            "goal": "maximize",
        },
        "parameters": {
            "LR": {
                "values": [
                    0.001,
                    0.0005,
                    0.0001,
                    0.00005,
                ]
            },
        },
    }

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(
        sweep_id, functools.partial(wrapped_make_train, default_config), count=1000
    )


def wrapped_make_train(default_config: Config):
    wandb.init(project=default_config["PROJECT"])

    config = copy.deepcopy(default_config)
    for k, v in dict(wandb.config).items():
        config[k] = v

    print("running experiment with params:", config)

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config)))
    outs = jax.block_until_ready(train_vjit(rngs))
    return outs


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    if config["HYP_TUNE"]:
        tune(config)
    else:
        single_run(config)


if __name__ == "__main__":
    main()
