import pytest
from dc_env.dc_scheduling_env import TaskSchedulingEnv
from simulation.datacenter_cluster_manager import DatacenterClusterManager

def test_environment_runs():
    from train_rl_agent import make_env
    env = make_env()
    obs, _ = env.reset(seed=123)
    assert isinstance(obs, list)
    if len(obs) > 0:
        assert isinstance(obs[0], list) or isinstance(obs[0], float)
