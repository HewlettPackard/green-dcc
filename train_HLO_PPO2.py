import os
import random
import numpy as np
import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.utils.filter import MeanStdFilter
from envs.heirarchical_env_cont import HeirarchicalDCRL, DEFAULT_CONFIG
from utils.create_trainable import create_wrapped_trainable

NUM_WORKERS = 4
RESULTS_DIR = './results/'

def sample_hyperparameters():
    """Sample a set of random hyperparameters."""
    gamma = random.uniform(0.3, 0.99)
    lr = 10**random.uniform(-6, -3)  # Random learning rate between 1e-6 and 1e-3
    kl_coeff = random.uniform(0.1, 0.5)
    clip_param = random.uniform(0.1, 0.3)
    entropy_coeff = random.uniform(0.0, 0.05)
    train_batch_size = random.choice([1024, 2048, 4096])
    sgd_minibatch_size = random.choice([64, 128, 256])
    num_sgd_iter = random.choice([10, 20, 30])
    hidden_layers = random.choice([[64], [64, 64], [128, 128], [256, 256], [256, 64, 16], [64, 64, 64], [64, 16, 4]])

    return {
        "gamma": gamma,
        "lr": lr,
        "kl_coeff": kl_coeff,
        "clip_param": clip_param,
        "entropy_coeff": entropy_coeff,
        "train_batch_size": train_batch_size,
        "sgd_minibatch_size": sgd_minibatch_size,
        "num_sgd_iter": num_sgd_iter,
        "fcnet_hiddens": hidden_layers
    }

if __name__ == '__main__':
    os.environ["RAY_DEDUP_LOGS"] = "0"
    ray.init(num_cpus=NUM_WORKERS+1, ignore_reinit_error=True)

    for i in range(1000):  # Run 10 different experiments with random hyperparameters
        # Sample hyperparameters
        hyperparams = sample_hyperparameters()

        # Update the config with the sampled hyperparameters
        config = (
            PPOConfig()
            .environment(
                env=HeirarchicalDCRL,
                env_config=DEFAULT_CONFIG
            )
            .framework("torch")
            .rollouts(
                num_rollout_workers=NUM_WORKERS,
            )
            .training(
                gamma=hyperparams["gamma"],
                lr=hyperparams["lr"],
                kl_coeff=hyperparams["kl_coeff"],
                clip_param=hyperparams["clip_param"],
                entropy_coeff=hyperparams["entropy_coeff"],
                train_batch_size=hyperparams["train_batch_size"],
                sgd_minibatch_size=hyperparams["sgd_minibatch_size"],
                num_sgd_iter=hyperparams["num_sgd_iter"],
                model={'fcnet_hiddens': hyperparams["fcnet_hiddens"]}
            )
            .resources(num_gpus=0)
            .debugging(seed=10)
        )

        # Create a unique experiment name with hyperparameters
        experiment_name = (
            f"PPO_gamma{hyperparams['gamma']:.2f}_lr{hyperparams['lr']:.1e}_"
            f"kl{hyperparams['kl_coeff']:.2f}_clip{hyperparams['clip_param']:.2f}_"
            f"ent{hyperparams['entropy_coeff']:.3f}_batch{hyperparams['train_batch_size']}_"
            f"minibatch{hyperparams['sgd_minibatch_size']}_sgd{hyperparams['num_sgd_iter']}_"
            f"fcnet{'x'.join(map(str, hyperparams['fcnet_hiddens']))}"
        )

        tune.Tuner(
            create_wrapped_trainable(PPO),
            param_space=config.to_dict(),
            run_config=air.RunConfig(
                stop={"training_iteration": 1000},
                verbose=0,
                local_dir=RESULTS_DIR,
                name=experiment_name,  # Use the experiment name
                checkpoint_config=ray.air.CheckpointConfig(
                    checkpoint_frequency=5,
                    num_to_keep=5,
                    checkpoint_score_attribute="episode_reward_mean",
                    checkpoint_score_order="max"
                ),
            )
        ).fit()

        # Optionally, log the hyperparameters to a file
        with open(os.path.join(RESULTS_DIR, experiment_name, "hyperparameters.txt"), "w") as f:
            for key, value in hyperparams.items():
                f.write(f"{key}: {value}\n")
