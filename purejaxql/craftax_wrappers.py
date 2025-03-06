import dataclasses
from functools import partial
from typing import Self

import chex
import jax
import jax.numpy as jnp
from craftax.craftax.craftax_state import StaticEnvParams
from gymnax.environments.environment import Environment, EnvState
from gymnax.wrappers.purerl import GymnaxWrapper
from gymnax.wrappers.purerl import LogEnvState as _LogEnvState
from jaxtyping import Float, Int
from xtils.jitpp import Static, jit


@dataclasses.dataclass(frozen=True)
class LogEnvState(_LogEnvState):
    # NOTE: This is an added field compared to gymnax's LogEnvState.
    # Also, this is usually in the base `EnvState` of gymnax, but isn't in the craftax
    # EnvState. (why?).
    timestep: int


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: Environment):
        super().__init__(env)
        self._env: Environment

    @jit
    def reset(
        self: Static[Self],
        key: chex.PRNGKey,
        params: Static[StaticEnvParams | None] = None,
    ) -> tuple[jax.Array, LogEnvState]:
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0.0, 0, 0.0, 0, 0)
        return obs, state

    @jit
    def step(
        self: Static[Self],
        key: chex.PRNGKey,
        state: LogEnvState,
        action: int | float,
        params: Static[StaticEnvParams | None] = None,
    ) -> tuple[jax.Array, LogEnvState, jax.Array, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info


@dataclasses.dataclass
class BatchLogEnvState:
    env_state: EnvState
    episode_returns: Float[jax.Array, " num_envs"]
    episode_lengths: Int[jax.Array, " num_envs"]
    returned_episode_returns: Float[jax.Array, " num_envs"]
    returned_episode_lengths: Int[jax.Array, " num_envs"]

    # NOTE: This is a tiny bit different than in gymnax.
    timestep: Int[jax.Array, " num_envs"]


class BatchEnvWrapper(GymnaxWrapper):
    """Batches reset and step functions"""

    def __init__(self, env: LogWrapper, num_envs: int):
        super().__init__(env)
        self._env: LogWrapper

        self.num_envs = num_envs

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @jit
    def reset(
        self: Static[Self],
        rng: chex.PRNGKey,
        params: Static[StaticEnvParams | None] = None,
    ) -> tuple[jax.Array, BatchLogEnvState]:
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @jit
    def step(
        self: Static[Self],
        rng: chex.PRNGKey,
        state: LogEnvState,
        action: jax.Array,
        params: Static[StaticEnvParams | None] = None,
    ):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, state, reward, done, info = self.step_fn(rngs, state, action, params)

        return obs, state, reward, done, info


class AutoResetEnvWrapper(GymnaxWrapper):
    """Provides standard auto-reset functionality, providing the same behaviour as Gymnax-default."""

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, params=None):
        return self._env.reset(key, params)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        rng, _rng = jax.random.split(rng)
        obs_st, state_st, reward, done, info = self._env.step(
            _rng, state, action, params
        )

        rng, _rng = jax.random.split(rng)
        obs_re, state_re = self._env.reset(_rng, params)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree.map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.lax.select(done, obs_re, obs_st)

            return obs, state

        obs, state = auto_reset(done, state_re, state_st, obs_re, obs_st)

        return obs, state, reward, done, info


class OptimisticResetVecEnvWrapper(GymnaxWrapper):
    """
    Provides efficient 'optimistic' resets.
    The wrapper also necessarily handles the batching of environment steps and resetting.
    reset_ratio: the number of environment workers per environment reset.  Higher means more efficient but a higher
    chance of duplicate resets.
    """

    def __init__(self, env: LogWrapper, num_envs: int, reset_ratio: int):
        super().__init__(env)

        self.num_envs = num_envs
        self.reset_ratio = reset_ratio
        assert num_envs % reset_ratio == 0, (
            "Reset ratio must perfectly divide num envs."
        )
        self.num_resets = self.num_envs // reset_ratio

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng: chex.PRNGKey, state: LogEnvState, action, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs_st, state_st, reward, done, info = self.step_fn(rngs, state, action, params)

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_resets)
        obs_re, state_re = self.reset_fn(rngs, params)

        rng, _rng = jax.random.split(rng)
        reset_indexes = jnp.arange(self.num_resets).repeat(self.reset_ratio)
        # TODO: There is this warning in jax.random.choice:
        # If p has fewer non-zero elements than the requested number of samples,
        # as specified in shape, and replace=False, the output of this function
        # is ill-defined. Please make sure to use appropriate inputs.
        being_reset = jax.random.choice(
            _rng,
            jnp.arange(self.num_envs),
            shape=(self.num_resets,),
            p=done,
            replace=False,
        )
        reset_indexes = reset_indexes.at[being_reset].set(jnp.arange(self.num_resets))

        obs_re = obs_re[reset_indexes]
        state_re = jax.tree.map(lambda x: x[reset_indexes], state_re)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree.map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.lax.select(done, obs_re, obs_st)

            return state, obs

        state, obs = jax.vmap(auto_reset)(done, state_re, state_st, obs_re, obs_st)

        return obs, state, reward, done, info
