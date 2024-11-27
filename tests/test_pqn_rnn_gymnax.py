import purejaxql
import purejaxql.pqn_rnn_gymnax
import purejaxql.pqn_rnn_gymnax_flat

from .conftest import (
    AlgoTests,
    use_alg_configs,
    use_fewer_timesteps,
)

# note: default number of timesteps in the config is 5e5, we make this shorter for tests to stay fast.


@use_alg_configs(["pqn_rnn_cartpole", "pqn_rnn_memory_chain"])
@use_fewer_timesteps(
    num_updates=2,
    num_steps=64,
    num_envs=32,
    total_timesteps=32 * 64 * 2,
)
class TestFlattenedPqnRNNGymnax(
    AlgoTests,
):  # ComparisonTests):
    # original_make_train = purejaxql.pqn_rnn_gymnax.make_train
    make_train = purejaxql.pqn_rnn_gymnax_flat.make_train
