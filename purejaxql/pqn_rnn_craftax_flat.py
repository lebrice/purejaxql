"""
This script uses BatchRenorm for more effective batch normalization in long training runs.
"""

import copy
import os
import time
from functools import partial
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import chex
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from craftax.craftax_env import make_craftax_env_from_name
from craftax_wrappers import BatchEnvWrapper, LogWrapper, OptimisticResetVecEnvWrapper
from flax.core.frozen_dict import FrozenDict
from flax.linen.module import Module, compact, merge_param
from flax.linen.normalization import _canonicalize_axes, _compute_stats, _normalize
from flax.training.train_state import TrainState
from flax.traverse_util import flatten_dict, unflatten_dict
from gymnax import EnvParams, EnvState
from jax.nn import initializers
from omegaconf import OmegaConf
from safetensors.flax import load_file, save_file
from xtils.jitpp import Static, jit

import wandb

PRNGKey = Any
Array = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Axes = Union[int, Sequence[int]]


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
    def __call__(self, x, use_running_average: Optional[bool] = None):
        use_running_average = merge_param(
            "use_running_average", self.use_running_average, use_running_average
        )
        feature_axes = _canonicalize_axes(x.ndim, self.axis)
        reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)
        feature_shape = [x.shape[ax] for ax in feature_axes]

        ra_mean = self.variable(
            "batch_stats",
            "mean",
            lambda s: jnp.zeros(s, jnp.float32),
            feature_shape,
        )
        ra_var = self.variable(
            "batch_stats", "var", lambda s: jnp.ones(s, jnp.float32), feature_shape
        )

        r_max = self.variable(
            "batch_stats",
            "r_max",
            lambda s: s,
            3,
        )
        d_max = self.variable(
            "batch_stats",
            "d_max",
            lambda s: s,
            5,
        )
        steps = self.variable(
            "batch_stats",
            "steps",
            lambda s: s,
            0,
        )

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
    def __call__(self, carry, x):
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
    norm_type: str = "layer_norm"
    dueling: bool = False
    add_last_action: bool = False

    @nn.compact
    def __call__(self, hidden, x, done, last_action, train: bool = False):
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
        for i in range(self.num_rnn_layers):
            rnn_in = (x, done)
            hidden_aux, x = ScannedRNN()(hidden[i], rnn_in)
            new_hidden.append(hidden_aux)

        q_vals = nn.Dense(self.action_dim)(x)

        return new_hidden, q_vals

    def initialize_carry(self, *batch_size):
        return [
            ScannedRNN.initialize_carry(self.hidden_size, *batch_size)
            for _ in range(self.num_rnn_layers)
        ]


@chex.dataclass(frozen=True)
class Transition:
    last_hs: chex.Array
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    last_done: chex.Array
    last_action: chex.Array
    q_vals: chex.Array


class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def make_train(config):
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

    # epsilon-greedy exploration
    return lambda rng: train(
        rng,
        env=env,
        env_params=env_params,
        test_env=test_env,
        config=FrozenDict(config),
    )


@jit
def train(
    rng: chex.PRNGKey,
    env: Static[OptimisticResetVecEnvWrapper | BatchEnvWrapper],
    env_params: Static[EnvParams],
    test_env: Static[OptimisticResetVecEnvWrapper | BatchEnvWrapper],
    config: Static[dict],
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
    # create_agent_fn = lambda rng: create_agent(
    rng, _rng = jax.random.split(rng)
    train_state = create_agent(
        rng,
        env=env,
        env_params=env_params,
        network=network,
        lr=lr,
        max_grad_norm=config["MAX_GRAD_NORM"],
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

    rng, _rng = jax.random.split(rng)
    obs, env_state = env.reset(_rng, env_params)
    init_dones = jnp.zeros((config["NUM_ENVS"]), dtype=bool)
    init_action = jnp.zeros((config["NUM_ENVS"]), dtype=int)
    init_hs = network.initialize_carry(config["NUM_ENVS"])
    expl_state = (init_hs, obs, init_dones, init_action, env_state)

    # step randomly to have the initial memory window
    rng, _rng = jax.random.split(rng)
    (*expl_state, rng), memory_transitions = jax.lax.scan(
        lambda *expl_state_and_rng: _random_step(
            *expl_state_and_rng,
            train_state=train_state,
            env=env,
            env_params=env_params,
            network=network,
            config=config,
        ),
        (*expl_state, _rng),
        None,
        config["MEMORY_WINDOW"] + config["NUM_STEPS"],
    )
    expl_state = tuple(expl_state)

    # train
    rng, _rng = jax.random.split(rng)
    runner_state = (train_state, memory_transitions, expl_state, test_metrics, _rng)

    runner_state, metrics = jax.lax.scan(
        lambda _runner_state, _update_index: _update_step(
            _runner_state,
            _update_index,
            network=network,
            config=config,
            env=env,
            test_env=test_env,
            env_params=env_params,
            eps_scheduler=eps_scheduler,
            original_rng=original_rng,
        ),
        runner_state,
        None,  # note: could also be jnp.arange(NUM_UDPATES)
        config["NUM_UPDATES"],
    )

    return {"runner_state": runner_state, "metrics": metrics}


@jit
def get_test_metrics(
    train_state: CustomTrainState,
    rng: chex.PRNGKey,
    *,
    config: Static[dict],
    network: Static[RNNQNetwork],
    test_env: Static[OptimisticResetVecEnvWrapper | BatchEnvWrapper],
    env_params: Static[EnvParams],
):
    if not config.get("TEST_DURING_TRAINING", False):
        return None

    def _greedy_env_step(step_state, _unused_step_index: jax.Array):
        hs, last_obs, last_done, last_action, env_state, rng = step_state
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
        new_obs, new_env_state, reward, new_done, info = test_env.step(
            _rng, env_state, new_action, env_params
        )
        step_state = (new_hs, new_obs, new_done, new_action, new_env_state, rng)
        return step_state, info

    rng, _rng = jax.random.split(rng)
    init_obs, env_state = test_env.reset(_rng, env_params)
    init_done = jnp.zeros((config["TEST_NUM_ENVS"]), dtype=bool)
    init_action = jnp.zeros((config["TEST_NUM_ENVS"]), dtype=int)
    init_hs = network.initialize_carry(config["TEST_NUM_ENVS"])  # (n_envs, hs_size)
    step_state = (
        init_hs,
        init_obs,
        init_done,
        init_action,
        env_state,
        _rng,
    )
    step_state, infos = jax.lax.scan(
        _greedy_env_step, step_state, None, config["TEST_NUM_STEPS"]
    )
    # return mean of done infos
    done_infos = jax.tree.map(
        lambda x: (x * infos["returned_episode"]).sum()
        / infos["returned_episode"].sum(),
        infos,
    )
    return done_infos


@jit
def _random_step(
    carry: tuple[Any, jax.Array, jax.Array, jax.Array, EnvState, chex.PRNGKey],
    _unused_step_index: jax.Array,
    *,
    train_state: CustomTrainState,
    network: Static[RNNQNetwork],
    config: Static[dict],
    env: Static[OptimisticResetVecEnvWrapper | BatchEnvWrapper],
    env_params: Static[EnvParams],
):
    hs, last_obs, last_done, last_action, env_state, rng = carry
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
    return (
        new_hs,
        new_obs,
        new_done,
        new_action,
        new_env_state,
        rng,
    ), transition


@jit
def _update_step(
    runner_state: tuple[CustomTrainState, Transition, tuple, dict, chex.PRNGKey],
    _unused_update_index: jax.Array,
    *,
    network: Static[RNNQNetwork],
    config: Static[dict],
    eps_scheduler: Static[optax.Schedule],
    env: Static[OptimisticResetVecEnvWrapper | BatchEnvWrapper],
    test_env: Static[OptimisticResetVecEnvWrapper | BatchEnvWrapper],
    env_params: Static[EnvParams],
    original_rng: jax.Array,
) -> tuple[tuple[CustomTrainState, Transition, tuple, dict, chex.PRNGKey], dict]:
    train_state, memory_transitions, expl_state, test_metrics, rng = runner_state

    # SAMPLE PHASE
    # step the env
    rng, _rng = jax.random.split(rng)
    (*expl_state, rng), (transitions, infos) = jax.lax.scan(
        lambda *expl_state_and_rng: _step_env(
            *expl_state_and_rng,
            train_state=train_state,
            env=env,
            env_params=env_params,
            eps_scheduler=eps_scheduler,
            config=config,
            network=network,
        ),
        (*expl_state, _rng),
        None,
        config["NUM_STEPS"],
    )
    expl_state = tuple(expl_state)

    train_state = train_state.replace(
        timesteps=train_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"]
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
        lambda train_state_and_rng, _epoch_index: _learn_epoch(
            train_state_and_rng,
            _epoch_index,
            memory_transitions=memory_transitions,
            network=network,
            config=config,
        ),
        (train_state, rng),
        None,
        config["NUM_EPOCHS"],
    )

    train_state = train_state.replace(n_updates=train_state.n_updates + 1)
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

    runner_state = (
        train_state,
        memory_transitions,
        tuple(expl_state),
        test_metrics,
        rng,
    )

    return runner_state, None


@jit
def _learn_epoch(
    carry: tuple[CustomTrainState, chex.PRNGKey],
    _unused_epoch_index: jax.Array,
    *,
    memory_transitions: Transition,
    network: Static[RNNQNetwork],
    config: Static[dict],
):
    train_state, rng = carry

    def preprocess_transition(x, rng):
        # x: (num_steps, num_envs, ...)
        x = jax.random.permutation(rng, x, axis=1)  # shuffle the transitions
        x = x.reshape(
            x.shape[0], config["NUM_MINIBATCHES"], -1, *x.shape[2:]
        )  # num_steps, minibatches, batch_size/num_minbatches,
        x = jnp.swapaxes(
            x, 0, 1
        )  # (minibatches, num_steps, batch_size/num_minbatches, ...)
        return x

    rng, _rng = jax.random.split(rng)
    minibatches = jax.tree_util.tree_map(
        lambda x: preprocess_transition(x, _rng),
        memory_transitions,
    )  # num_minibatches, num_steps+memory_window, batch_size/num_minbatches, ...

    rng, _rng = jax.random.split(rng)
    (train_state, rng), (loss, qvals) = jax.lax.scan(
        lambda train_state_and_rng, minibatch: _learn_phase(
            train_state_and_rng,
            minibatch,
            network=network,
            config=config,
        ),
        (train_state, rng),
        minibatches,
    )

    return (train_state, rng), (loss, qvals)


@jit
def _learn_phase(
    carry: tuple[CustomTrainState, chex.PRNGKey],
    minibatch: Transition,
    *,
    network: Static[RNNQNetwork],
    config: Static[dict],
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
            minibatch=minibatch,
            train_state=train_state,
            agent_in=agent_in,
            hs=hs,
            network=network,
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
    params: dict[str, jax.Array],
    *,
    minibatch: Transition,
    train_state: CustomTrainState,
    agent_in: tuple[jax.Array, jax.Array, jax.Array],
    hs: Any,
    network: Static[RNNQNetwork],
    config: Static[dict],
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
    *,
    config: Static[dict],
):
    def _get_target(lambda_returns_and_next_q, rew_q_done):
        reward, q, done = rew_q_done
        lambda_returns, next_q = lambda_returns_and_next_q
        target_bootstrap = reward + config["GAMMA"] * (1 - done) * next_q
        delta = lambda_returns - next_q
        lambda_returns = target_bootstrap + config["GAMMA"] * config["LAMBDA"] * delta
        lambda_returns = (1 - done) * lambda_returns + done * reward
        next_q = jnp.max(q, axis=-1)
        return (lambda_returns, next_q), lambda_returns

    lambda_returns = reward[-1] + config["GAMMA"] * (1 - done[-1]) * last_q
    last_q = jnp.max(q_vals[-1], axis=-1)
    _, targets = jax.lax.scan(
        _get_target,
        (lambda_returns, last_q),
        jax.tree.map(lambda x: x[:-1], (reward, q_vals, done)),
        reverse=True,
    )
    targets = jnp.concatenate([targets, lambda_returns[np.newaxis]])
    return targets


@jit
def _step_env(
    carry: tuple[Any, jax.Array, jax.Array, jax.Array, EnvState, chex.PRNGKey],
    _unused_step_index: jax.Array,
    *,
    train_state: CustomTrainState,
    env: Static[OptimisticResetVecEnvWrapper | BatchEnvWrapper],
    env_params: Static[EnvParams],
    network: Static[RNNQNetwork],
    config: Static[dict],
    eps_scheduler: Static[optax.Schedule],
):
    hs, last_obs, last_done, last_action, env_state, rng = carry
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
    return (new_hs, new_obs, new_done, new_action, new_env_state, rng), (
        transition,
        info,
    )


def eps_greedy_exploration(rng: jax.Array, q_vals: jax.Array, eps: jax.Array):
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


@jit
def create_agent(
    rng: chex.PRNGKey,
    env: Static[OptimisticResetVecEnvWrapper | BatchEnvWrapper],
    env_params: EnvParams,
    network: Static[RNNQNetwork],
    lr: Static[float | optax.Schedule],
    max_grad_norm: Static[float],
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
        optax.clip_by_global_norm(max_grad_norm),
        optax.radam(learning_rate=lr),
    )

    train_state = CustomTrainState.create(
        apply_fn=network.apply,
        params=network_variables["params"],
        batch_stats=network_variables["batch_stats"],
        tx=tx,
    )
    return train_state


def single_run(config):
    config = {**config, **config["alg"]}

    alg_name = config.get("ALG_NAME", "pqn_rnn")
    env_name = config["ENV_NAME"]

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
        name=f"{config['ALG_NAME']}_{config['ENV_NAME']}",
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])

    t0 = time.time()
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config))).lower(rngs).compile()
    print(f"Took {time.time() - t0} seconds to JIT.")
    t0 = time.time()
    outs = jax.block_until_ready(train_vjit(rngs))
    print(f"Took {time.time() - t0} seconds to complete.")

    if config.get("SAVE_PATH", None) is not None:
        model_state = outs["runner_state"][0]
        save_dir = os.path.join(config["SAVE_PATH"], env_name)
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


def tune(default_config):
    """Hyperparameter sweep with wandb."""

    default_config = {**default_config, **default_config["alg"]}
    alg_name = default_config.get("ALG_NAME", "pqn")
    env_name = default_config["ENV_NAME"]

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])

        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            config[k] = v

        print("running experiment with params:", config)

        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config)))
        outs = jax.block_until_ready(train_vjit(rngs))

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
    wandb.agent(sweep_id, wrapped_make_train, count=1000)


def save_params(params: dict, filename: str | os.PathLike) -> None:
    flattened_dict = flatten_dict(params, sep=",")
    save_file(flattened_dict, filename)  # type: ignore


def load_params(filename: str | os.PathLike) -> dict:
    flattened_dict = load_file(filename)
    return unflatten_dict(flattened_dict, sep=",")


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
