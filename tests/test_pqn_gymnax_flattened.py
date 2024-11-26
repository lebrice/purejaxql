import operator
from typing import Any, Mapping
from purejaxql.pqn_gymnax_flattened import make_train
import jax
import pytest
from hydra import initialize, compose
import numpy as np
from purejaxql.pqn_gymnax_flattened import Config
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture
from flax.traverse_util import flatten_dict


@pytest.fixture
def command_line_overrides(request: pytest.FixtureRequest) -> list[str]:
    return list(getattr(request, "param", []))


# note: default number of timesteps in the config is 5e5, making it shorter to keep tests fast.
use_fewer_timesteps = pytest.mark.parametrize(
    command_line_overrides.__name__,
    [["alg.TOTAL_TIMESTEPS=100", "alg.TOTAL_TIMESTEPS_DECAY=100"]],
    indirect=True,
)


@pytest.fixture
def config(command_line_overrides: list[str]):
    with initialize(version_base="1.2", config_path="../purejaxql/config"):
        overrides = ["+alg=pqn_cartpole", *command_line_overrides]
        print(
            "Running test with the same config as if this was passed on the command-line:"
        )
        print(f"python purejaxql/pqn_gymnax_flattened.py {' '.join(overrides)}")
        _config = compose(
            config_name="config",
            overrides=overrides,
        )
        config: Config = {**_config, **_config["alg"]}  # type: ignore
        yield config


@pytest.fixture(params=[42, 123])
def seed(request: pytest.FixtureRequest) -> int:
    return getattr(request, "param")


@pytest.fixture(params=[None, 5])
def num_seeds(request: pytest.FixtureRequest) -> int | None:
    return getattr(request, "param")


@pytest.fixture(params=[True, False], ids=["jit", "no_jit"])
def jit(request: pytest.FixtureRequest) -> bool:
    return getattr(request, "param")


@use_fewer_timesteps
def test_train_is_deterministic(
    config: Config, jit: bool, seed: int, num_seeds: int | None
):
    train_fn = make_train(config)
    if num_seeds is not None:
        train_fn = jax.vmap(train_fn)
    if jit:
        train_fn = jax.jit(train_fn)

    rng = jax.random.PRNGKey(seed)
    if num_seeds is not None:
        rng = jax.random.split(rng, num_seeds)

    outputs_1 = train_fn(rng)
    outputs_2 = train_fn(rng)
    jax.block_until_ready((outputs_1, outputs_2))
    jax.tree.map(np.testing.assert_allclose, outputs_1, outputs_2)


@use_fewer_timesteps
def test_train_is_reproducible(
    ndarrays_regression: NDArraysRegressionFixture,
    config: Config,
    jit: bool,
    seed: int,
    num_seeds: int | None,
):
    """Test that the results of `train` are reproduble for the same seed given the same hardware config."""
    train_fn = make_train(config)
    if num_seeds is not None:
        train_fn = jax.vmap(train_fn)
    if jit:
        train_fn = jax.jit(train_fn)

    rng = jax.random.PRNGKey(seed)
    if num_seeds is not None:
        rng = jax.random.split(rng, num_seeds)

    outputs = train_fn(rng)
    print(jax.tree.map(np.shape, outputs))
    # TODO: there seems to be a tuple somewhere in the runner_state entry of the outputs.
    ndarrays_regression.check(
        flatten_dict(
            jax.tree.map(operator.methodcaller("__array__"), outputs["metrics"]),
            sep=".",
        )
    )


def convert_list_and_tuples_to_dicts(value: Any) -> Any:
    """Converts all lists and tuples in a nested structure to dictionaries.

    >>> convert_list_and_tuples_to_dicts([1, 2, 3])
    {'0': 1, '1': 2, '2': 3}
    >>> convert_list_and_tuples_to_dicts((1, 2, 3))
    {'0': 1, '1': 2, '2': 3}
    >>> convert_list_and_tuples_to_dicts({"a": [1, 2, 3], "b": (4, 5, 6)})
    {'a': {'0': 1, '1': 2, '2': 3}, 'b': {'0': 4, '1': 5, '2': 6}}
    """
    if isinstance(value, Mapping):
        return {k: convert_list_and_tuples_to_dicts(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        # NOTE: Here we won't be able to distinguish between {"0": "bob"} and ["bob"]!
        # But that's not too bad.
        return {
            f"{i}": convert_list_and_tuples_to_dicts(v) for i, v in enumerate(value)
        }
    return value
