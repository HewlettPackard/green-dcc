import os
import pytest
from train_rl_agent import make_env

def test_environment_runs():
    sim_path = "configs/env/sim_config.yaml"
    dc_path = "configs/env/datacenters.yaml"
    reward_path = "configs/env/reward_config.yaml"

    env = make_env(sim_path, dc_path, reward_path, writer=None)
    obs, _ = env.reset(seed=123)

    assert isinstance(obs, list)
    if len(obs) > 0:
        assert isinstance(obs[0], list) or isinstance(obs[0], float)
