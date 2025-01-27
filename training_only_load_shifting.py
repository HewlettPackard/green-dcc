#%%
import torch
import random
import string
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch.nn as nn

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

# Import your custom environment (the code you posted)
# Make sure your directory structure / imports allow this:
from envs.dcrl_env_harl_partialobs_sb3 import DCRL, EnvConfig

from utils.sb_callbacks import CO2WaterUsageCallback, TemporalLoadShiftingCallback


from custom_policies import ScalableAttentionPolicy, ScalableTransformerFeatureExtractor, TransformerFeatureExtractor, TransformerActorCriticPolicy

#%%

def make_env_ls_only(config, rank):
    """
    Utility function for creating a single instance of your load-shifting environment.
    The 'rank' can be used to set different seeds for each env.
    """
    def _init():
        # Update the config for single-agent setup
        ls_only_config = config.copy()
        ls_only_config["agents"] = ["agent_ls"]  # Only load-shifting agent active
        ls_only_config['flexible_load'] = 0.4
        ls_only_config['location'] = 'ca'
        ls_only_config['month'] = 6
        ls_only_config['initialize_queue_at_reset'] = True
        ls_only_config['random_init_day_at_reset'] = True
        # ls_only_config['normalization_file'] = '/lustre/guillant/green-dcc/vec_normalize_56JL3O.pkl'
        
        env = DCRL(ls_only_config)  # Pass the modified config here
        env.set_seed(rank)  # Set the seed
        # Wrap each environment with Monitor to record episode stats
        env = Monitor(env)
        return env
    return _init

def my_custom_lr_schedule(progress_remaining: float) -> float:
    # Example: exponential decay
    # LR goes from 3e-4 to 3e-6
    return 3e-4 * (3e-6 / 3e-4) ** (1 - progress_remaining)
#%%

# Import your environment class
# from my_env_file import HeirarchicalDCRL, DEFAULT_CONFIG  # Adjust this import path

if __name__ == "__main__":
    # Number of parallel environments
    n_envs = 64  # Adjust as needed for computational resources

    config = EnvConfig.DEFAULT_CONFIG.copy()

    # Create a list of environment-building functions
    env_fns = [make_env_ls_only(config, i) for i in range(n_envs)]

    # Build SubprocVecEnv (runs each env in its own process)
    vec_env = SubprocVecEnv(env_fns)
    
    # Apply the FrameStack wrapper
    # vec_env = VecFrameStack(vec_env, n_stack=4)
    
    # Wrap the environment with VecNormalize for observation normalization
    # If you want to normalize actions (for continuous action spaces), set `norm_obs=False` and `norm_reward=False` as needed
    # vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 2) Construct policy_kwargs to use our custom feature extractor
    policy_kwargs = dict(
        # Define network architecture for the actor and critic
        net_arch=dict(
            pi=[64, 64],  # Policy network layers
            vf=[64, 64]   # Value network layers
        ),
        activation_fn=nn.ReLU,
        # activation_fn=nn.Tanh,
    )
    # policy_kwargs = dict(
    # net_arch=[dict(pi=[64, 64], vf=[64, 64])],  # Policy and Value networks
    # activation_fn=nn.ReLU,                     # Activation function
    # n_lstm_layers=2,                           # Number of LSTM layers
    # lstm_hidden_size=128,                      # Size of LSTM hidden state
    # shared_lstm=False                           # Share LSTM across policy and value networks
    # )


    # 3) Instantiate the model
    model = PPO(
        policy="MlpPolicy",     # We'll override the default MLP with our custom extractor
        env=vec_env,
        device="cuda:0",             # <---- Force using GPU if available
        # device="cpu",
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=512,            # Number of steps to run for each environment per update
        batch_size=512,
        learning_rate=1e-4,  # Custom learning rate schedule
        gamma=0.995,
        ent_coef=0.05,
        tensorboard_log="./tb_logs/",   # <--- directory for TensorBoard logs
        clip_range=0.2,
        # ... any other hyperparameters ...
    )
    # model = RecurrentPPO(
    #     policy="MlpLstmPolicy",
    #     env=vec_env,
    #     device="cuda",              # Use GPU if available
    #     policy_kwargs=policy_kwargs,
    #     verbose=1,
    #     n_steps=512,                # Number of steps per update
    #     batch_size=256,
    #     learning_rate=1e-5,
    #     gamma=0.995,
    #     ent_coef=0.01,
    #     tensorboard_log="./tb_logs/",
    # )
    # Create a random string of 6 characters to the run name
    run_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    # Create a callback that saves the model every 10,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=f'./checkpoints/load_shifting/{run_name}',
        name_prefix='ls_ppo'
    )
    
    # Instantiate the custom callback
    custom_metrics_callback = TemporalLoadShiftingCallback(verbose=1)
    
    # 4) Train the model
    model.learn(
        total_timesteps=100_000_000,
        tb_log_name=f"ls_ppo_run_{run_name}",
        callback=[checkpoint_callback, custom_metrics_callback]
    )
    # 5) Save the model and VecNormalize statistics
    model.save(f"ls_ppo_model_{run_name}")
    # vec_env.save(f"vec_normalize_{run_name}.pkl")

    # vec_env.close()


#%%
