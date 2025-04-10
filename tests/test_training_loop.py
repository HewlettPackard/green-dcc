import os
import shutil
import tempfile
from train_rl_agent import train

def test_train_loop_runs_short():
    # Create temp dir and config with reduced steps
    with tempfile.TemporaryDirectory() as tmpdir:
        algo_cfg_path = os.path.join(tmpdir, "algorithm_config.yaml")
        with open(algo_cfg_path, "w") as f:
            f.write("""
                    algorithm:
                    gamma: 0.99
                    alpha: 0.01
                    actor_learning_rate: 1e-4
                    critic_learning_rate: 1e-4
                    batch_size: 32
                    tau: 0.005
                    replay_buffer_size: 1000
                    warmup_steps: 5
                    total_steps: 10
                    update_frequency: 1
                    policy_update_frequency: 2
                    save_interval: 20
                    hidden_dim: 32
                    max_tasks: 100
                    device: "cpu"
                    """)

    # Run train using config paths
    os.system(
        f"python train_rl_agent.py "
        f"--sim-config configs/env/sim_config.yaml "
        f"--dc-config configs/env/datacenters.yaml "
        f"--reward-config configs/env/reward_config.yaml "
        f"--algo-config {algo_cfg_path} "
    )
