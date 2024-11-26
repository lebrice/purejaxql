from purejaxql.pqn_gymnax import make_train
import jax
import time
import pytest
from hydra import initialize, compose
from omegaconf import DictConfig
import numpy as np
from purejaxql.pqn_gymnax import Config


@pytest.fixture
def config():
    with initialize(version_base="1.2", config_path="../purejaxql/config"):
        _config = compose(config_name="config", overrides=["+alg=pqn_cartpole"])
        config: Config = {**_config, **_config["alg"]}  # type: ignore
        yield config


@pytest.mark.parametrize("seed", [42, 123, 456])
@pytest.mark.parametrize("num_seeds", [1, 3])
def test_results_are_deterministic(seed: int, num_seeds: int, config: DictConfig):
    rngs = jax.random.split(jax.random.PRNGKey(seed), num_seeds)

    train_vjit = jax.jit(jax.vmap(make_train(config)))  # type: ignore
    outputs_1 = train_vjit(rngs)
    outputs_2 = train_vjit(rngs)
    jax.block_until_ready((outputs_1, outputs_2))
    jax.tree.map(np.testing.assert_allclose, outputs_1, outputs_2)
