import torch
from harl.envs.dcrl.dcrl_ptzoo import DCRLPettingZooEnv
from pettingzoo.utils.conversions import parallel_wrapper_fn

import gymnasium
from gymnasium import spaces

import numpy as np

import supersuit as ss

class DCRLEnv:
    def __init__(self, env_args):
        self.env_args = env_args
        self.env = DCRLPettingZooEnv(self.env_args)
        self.n_agents = len(self.env.env.agents)
        self.max_cycles = 25
        self.cur_step = 0
        
        # env1 = ls, env2 = dc, env3 = bat
        # The observation should be padded to have the same shape
        # Define observation spaces and action spaces according to the DCRL environment
        
        self.env = ss.pad_action_space_v0(ss.pad_observations_v0(self.env))
        
        self._seed = 0
        self.agents = self.env.possible_agents
        
        # if using flag if env_config['nonoverlapping_shared_obs_space']: do not unwrap the share_obs_space
        # if env_args['nonoverlapping_shared_obs_space']:
        #     self.share_observation_space = self.env.unwrapped.share_observation_space
        # else:
        self.share_observation_space = self.unwrap(self.env.unwrapped.share_observation_space)
        self.observation_space = self.unwrap(self.env.observation_spaces)
        self.action_space = self.unwrap(self.env.action_spaces)
        
        self.discrete = True

    def reset(self):
        # return obs, state, available_actions
        self._seed += 1
        self.cur_step = 0
        obs = self.unwrap(self.env.reset(seed=self._seed))
        
        states = tuple(o for o in obs)
        if self.env_args['nonoverlapping_shared_obs_space']:
            # these are hardcoded because the location/indices of these values are decided inside base environment currently
            # and not specified in the .yaml config for the dcrl env
            states =  np.concatenate((states[0][0:10], states[1][4:9], states[2][5].reshape(1,)), axis=None)
        else:
            states =  np.concatenate(states, axis=None)
        s_obs = self.repeat(states)
        
        avail_actions = self.get_avail_actions()
        return obs, s_obs, avail_actions
    
    def step(self, actions):
        # return obs, state, rewards, dones, info, available_actions
        
        # Shapes:
        # obs: [self.observation_space[agent_n].sample() for agent_n in self.n_agents]
        # state: [self.share_observation_space[agent_n].sample() for agent_n in self.n_agents]
        
        actions = self.wrap(actions.flatten())
        obs, rew, term, trunc, info = self.env.step(actions)
        
        obs = self.unwrap(obs)
        states = tuple(o for o in obs)
        if self.env_args['nonoverlapping_shared_obs_space']:
            # these are hardcoded because the location/indices of these values are decided inside base environment currently
            # and not specified in the .yaml config for the dcrl env
            states =  np.concatenate((states[0][0:10], states[1][4:9], states[2][5].reshape(1,)), axis=None)
        else:
            states =  np.concatenate(states, axis=None)
        s_obs = self.repeat(states)
        
        rewards = [[rew[agent]] for agent in self.agents]
        dones = {agent: term[agent] or trunc[agent] for agent in self.agents}
        
        return (
            obs,
            s_obs,
            rewards,
            self.unwrap(dones),
            self.unwrap(info),
            self.get_avail_actions(),
        )

    def seed(self, seed):
        self._seed = seed
        
    def close(self):
        pass
    
    def wrap(self, l):  # converts a list to a dict with agent names from base env
        d = {}
        for i, agent in enumerate(self.agents):
            d[agent] = l[i]
        return d

    def unwrap(self, d):  # converts a dict to a list with agent names from base env (agent names are queried from base env to preserve order)
        l = []
        for agent in self.agents:
            l.append(d[agent])
        return l

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]
    
    def get_avail_actions(self):
        if self.discrete:
            avail_actions = []
            for agent_id in range(self.n_agents):
                avail_agent = self.get_avail_agent_actions(agent_id)
                avail_actions.append(avail_agent)
            return avail_actions
        else:
            return None

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return [1] * self.action_space[agent_id].n