#%%
import torch
import random
import string
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

# Import your custom environment (the code you posted)
# Make sure your directory structure / imports allow this:
from envs.heirarchical_env_cont_random_location import HeirarchicalDCRL, DEFAULT_CONFIG
from utils.sb_callbacks import CO2WaterUsageCallback


from custom_policies import ScalableAttentionPolicy, ScalableTransformerFeatureExtractor, TransformerFeatureExtractor, TransformerActorCriticPolicy

#%%
def make_env(config, rank, random_locations=True):
    """
    Utility function for creating a single instance of your environment.
    The 'rank' can be used to set different seeds for each env.
    """
    def _init():
        env = HeirarchicalDCRL(config, random_locations=random_locations)  # Pass the flag here
        # Wrap each environment with Monitor to record episode stats
        env = Monitor(env)
        # env.seed(rank*100)  # or np.random.seed(rank)
        return env
    return _init

def my_custom_lr_schedule(progress_remaining: float) -> float:
    # Example: exponential decay
    # LR goes from 3e-4 to 3e-6
    return 3e-4 * (3e-6 / 3e-4) ** (1 - progress_remaining)
#%%
import gymnasium as gym
from stable_baselines3 import PPO

# Import your environment class
# from my_env_file import HeirarchicalDCRL, DEFAULT_CONFIG  # Adjust this import path

if __name__ == "__main__":
    # Number of parallel environments
    n_envs = 16
    
    # Set random_locations=True for training
    random_locations_training = True
    
    # Create a list of environment-building functions with random_locations=True
    env_fns = [make_env(DEFAULT_CONFIG, i, random_locations=random_locations_training) for i in range(n_envs)]

    # Build SubprocVecEnv (runs each env in its own process)
    vec_env = SubprocVecEnv(env_fns)
    
    # Wrap the environment with VecNormalize for observation normalization
    # If you want to normalize actions (for continuous action spaces), set `norm_obs=False` and `norm_reward=False` as needed
    # vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 2) Construct policy_kwargs to use our custom feature extractor
    policy_kwargs = dict(
        # Define network architecture for the actor and critic
        net_arch=dict(
            pi=[16, 8],  # Policy network layers
            vf=[16, 8]   # Value network layers
        ),
        activation_fn=nn.ReLU,
    )

    # 3) Instantiate the model
    model = PPO(
        policy="MlpPolicy",     # We'll override the default MLP with our custom extractor
        env=vec_env,
        device="cuda",             # <---- Force using GPU if available
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=64,            # Number of steps to run for each environment per update
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.0,
        ent_coef=0.01,
        tensorboard_log="./tb_logs/",   # <--- directory for TensorBoard logs
        # ... any other hyperparameters ...
    )
    # Create a random string of 6 characters to the run name
    run_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    # Create a callback that saves the model every 10,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=f'./checkpoints/{run_name}',
        name_prefix='transformer_ppo'
    )
    
    # Instantiate the custom callback
    custom_metrics_callback = CO2WaterUsageCallback(verbose=1)
    
    # 4) Train the model
    model.learn(
        total_timesteps=1_000_000,
        tb_log_name=f"transformer_ppo_run_{run_name}",
        callback=[checkpoint_callback, custom_metrics_callback]
    )
    # 5) Save the model and VecNormalize statistics
    model.save(f"transformer_ppo_model_{run_name}")
    # vec_env.save(f"vec_normalize_{run_name}.pkl")

    vec_env.close()


#%%
