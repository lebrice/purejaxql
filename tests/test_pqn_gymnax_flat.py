import pytest

import purejaxql
import purejaxql.pqn_gymnax
import purejaxql.pqn_gymnax_flat

from .conftest import BaseTests, use_alg_configs, use_fewer_timesteps


@use_alg_configs(["pqn_cartpole"])
@use_fewer_timesteps(
    num_updates=2,
    num_steps=64,
    num_envs=32,
    total_timesteps=32 * 64 * 2,
)
class TestFlattenedPqnGymnax(BaseTests):
    original_make_train = purejaxql.pqn_gymnax.make_train
    new_make_train = purejaxql.pqn_gymnax_flat.make_train

    @pytest.mark.parametrize(
        "jit",
        [
            True,
            pytest.param(
                False,
                marks=pytest.mark.xfail(
                    reason="TODO: results aren't precisely the same when jit=False."
                ),
            ),
        ],
        ids=["jit", "no_jit"],
    )
    def test_results_are_the_same(
        self,
        config: purejaxql.pqn_gymnax_flat.Config,
        jit: bool,
        seed: int,
        num_seeds: int | None,
    ):
        super().test_results_are_the_same(
            config=config, jit=jit, seed=seed, num_seeds=num_seeds
        )
