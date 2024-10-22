from tqdm import tqdm
from ray.rllib.env import MultiAgentEnv

from envs.heirarchical_env_cont import HeirarchicalDCRL, DEFAULT_CONFIG
# from tensorboardX import SummaryWriter

import gymnasium as gym
from gymnasium.spaces import Dict, Box
import numpy as np
import torch
import random

class TrulyHeirarchicalDCRL(HeirarchicalDCRL, MultiAgentEnv):

    def __init__(self, config):
        HeirarchicalDCRL.__init__(self, config)
        MultiAgentEnv.__init__(self)
        # self.writer = SummaryWriter("logs_single")
        self.global_step = 0
        
        # Perform a reset to initialize observations
        initial_obs, _ = self.reset()

        # Define per-agent observation spaces
        observation_spaces = {
            'high_level_policy': gym.spaces.Box(
                low=-10.0, high=10.0, shape=initial_obs['high_level_policy'].shape, dtype=np.float32
            )
        }

        for dc in self.datacenter_ids:
            observation_spaces[f"{dc}_ls_policy"] = gym.spaces.Box(
                low=-10.0, high=10.0, shape=initial_obs[f"{dc}_ls_policy"].shape, dtype=np.float32
            )

        self.observation_space = gym.spaces.Dict(observation_spaces)

        # Define per-agent action spaces
        # Assuming all agents share the same action space
        action_spaces = {
            'high_level_policy': self.action_space
        }

        for dc in self.datacenter_ids:
            action_spaces[f"{dc}_ls_policy"] = self.datacenters['DC1'].action_space[0]
            

        self.action_space = gym.spaces.Dict(action_spaces)
        
        # List of all possible agents that can ever appear in the environment
        self.possible_agents = ['high_level_policy'] + [f'{dc}_ls_policy' for dc in self.datacenter_ids]

        # List of agents active in the current episode (starts as possible agents)
        self.agents = self.possible_agents.copy()
        


    def reset(self, seed=None, options=None):
        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

        # Reset the base environment
        super().reset(seed)

        obs = {
            'high_level_policy': self.flat_obs.copy()
        }

        for dc in self.datacenter_ids:
            obs[f"{dc}_ls_policy"] = self.low_level_observations[dc]['agent_ls'].copy()

        return obs, {}
    
    def step(self, actions: dict):
        # Transform and enforce actions
        transformed_actions = self.transform_actions(actions['high_level_policy'])
        self.original_workload, self.overassigned_workload = self.safety_enforcement(transformed_actions)

        # Move workload within DCs
        low_level_actions = {
            dc: {'agent_ls': actions[f"{dc}_ls_policy"]} for dc in self.datacenter_ids
        }

        done = self.low_level_step(low_level_actions)

        # Prepare observations and rewards
        obs = {
            'high_level_policy': self.flat_obs.copy()
        }
        rewards = {
            'high_level_policy': self.calc_reward()
        }

        for dc in self.datacenter_ids:
            obs[f"{dc}_ls_policy"] = self.low_level_observations[dc]['agent_ls'].copy()
            rewards[f"{dc}_ls_policy"] = self.low_level_rewards[dc]['agent_ls']

        # Define termination and truncation
        terminated = {"__all__": False}
        truncated = {"__all__": done}

        if done:
            totalfp = sum(sum(self.metrics[dc]['bat_CO2_footprint']) for dc in self.datacenter_ids)
            print(f'The total CO2 footprint is {totalfp}')

            # Log the scalar totalfp to TensorBoard
            # self.writer.add_scalar("Total CO2 footprint", totalfp, self.global_step)
            # self.writer.flush()
            self.global_step += 1  # Increment the step counter

        # Prepare the info dictionary with custom metrics
        info = {}
        for dc in self.datacenter_ids:
            info[f"{dc}_ls_policy"] = self.low_level_infos[dc]
        return obs, rewards, terminated, truncated, info

    
if __name__ == '__main__':
    env = TrulyHeirarchicalDCRL(DEFAULT_CONFIG)

    done = False
    obs, _ = env.reset()
    
    with tqdm(total=env._max_episode_steps) as pbar:
        while not done:
            actions = {}
            actions['high_level_policy'] = env.action_space.sample()
            for dc in env.datacenter_ids:
                actions[dc + '_ls_policy'] = env.datacenters['DC1'].ls_env.action_space.sample()

            obs, rewards, terminated, truncated, info = env.step(actions)
            done = truncated['__all__']

            pbar.update(1)