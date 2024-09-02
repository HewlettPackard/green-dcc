import os
import random
import warnings

import torch
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete

from envs.dcrl_env_harl_partialobs import DCRL
from utils.hierarchical_workload_optimizer import WorkloadOptimizer
from utils.low_level_wrapper import LowLevelActorRLLIB, LowLevelActorHARL

warnings.filterwarnings(
    action="ignore",
    category=UserWarning
    )

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = {
    # DC1
    'config1' : {
        'location': 'ny',
        'cintensity_file': 'NY_NG_&_avgCI.csv',
        'weather_file': 'USA_NY_New.York-LaGuardia.epw',
        'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv',
        'dc_config_file': 'dc_config_dc1.json',
        'datacenter_capacity_mw' : 1.0,
        'timezone_shift': 8,
        'month': 7,
        'days_per_episode': 30,
        'partial_obs': True,
        'nonoverlapping_shared_obs_space': True,
        'debug': False,
        },

    # DC2
    'config2' : {
        'location': 'ga',
        'cintensity_file': 'GA_NG_&_avgCI.csv',
        'weather_file': 'USA_GA_Atlanta-Hartsfield-Jackson.epw',
        'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv',
        'dc_config_file': 'dc_config_dc1.json',
        'datacenter_capacity_mw' : 1.0,
        'timezone_shift': 0,
        'month': 7,
        'days_per_episode': 30,
        'partial_obs': True,
        'nonoverlapping_shared_obs_space': True,
        'debug': False,
        },

    # DC3
    'config3' : {
        'location': 'ca',
        'cintensity_file': 'CA_NG_&_avgCI.csv',
        'weather_file': 'USA_CA_San.Jose-Mineta.epw',
        'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv',
        'dc_config_file': 'dc_config_dc1.json',
        'datacenter_capacity_mw' : 1.0,
        'timezone_shift': 16,
        'month': 7,
        'days_per_episode': 30,
        'partial_obs': True,
        'nonoverlapping_shared_obs_space': True,
        'debug': False,
        },
    
    # Number of transfers per step
    'num_transfers': 2,

    # List of active low-level agents
    'active_agents': ['agent_dc'],

    # config for loading trained low-level agents
    'low_level_actor_config': {
        'harl': {
            'algo' : 'haa2c',
            'env' : 'dcrl',
            'exp_name' : 'll_actor',
            'model_dir': f'{CURR_DIR}/seed-00001-2024-05-01-21-50-12',
            },
        'rllib': {
            'checkpoint_path': f'{CURR_DIR}/maddpg/checkpoint_000000/',
            'is_maddpg': True
        }
    },
}

class HeirarchicalDCRL(gym.Env):

    def __init__(self, config: dict = DEFAULT_CONFIG):

        self.config = config
        self.penalty = 0#0.07
        # Init all datacenter environments
        DC1 = DCRL(config['config1'])
        DC2 = DCRL(config['config2'])
        DC3 = DCRL(config['config3'])

        self.datacenters = {
            'DC1': DC1,
            'DC2': DC2,
            'DC3': DC3,
        }

        self.datacenter_ids = list(self.datacenters.keys())
        
        # Load trained lower level agent
        self.lower_level_actor = LowLevelActorHARL(
            config['low_level_actor_config'],
            config['active_agents']
            )
        
        # self.lower_level_actor = LowLevelActorRLLIB(
        #     config['low_level_actor_config'], 
        #     config['active_agents']
        #     )
        
        # Set max episode steps
        self._max_episode_steps = 4 * 24 * DEFAULT_CONFIG['config1']['days_per_episode']
        self.max_episode_steps = 4 * 24 * DEFAULT_CONFIG['config1']['days_per_episode']
        # Add the spec attribute with max_episode_steps
        # self.spec = gym.spec(
        #     id="HeirarchicalDCRL-v0",
        #     max_episode_steps=self._max_episode_steps
        # )

        # Define observation and action space
        # List of observations that we get from each DC
        self.observations = [
            # 'dc_capacity',
            'curr_workload',
            'weather',
            # 'total_power_kw',
            'ci',
        ]
        # # This is the observation for each DC
        # self.dc_observation_space = Dict({obs: Box(-2, 2) for obs in self.observations})
        
        # # Observation space for this environment
        # self.observation_space = Dict({dc: self.dc_observation_space for dc in self.datacenters})
        
        # Number of datacenters
        num_dcs = len(self.datacenters)

        # Each observation component has a shape of (1,) so we need to multiply by the number of observations and datacenters
        observation_dim = len(self.observations) * num_dcs

        # Define continuous observation space for the flattened observation array
        self.observation_space = Box(
            low=-2.0, high=2.0, shape=(observation_dim,), dtype=np.float32
        )
        

        # Define continuous action space with three variables for transfers
        self.action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=float) # DC1-DC2, DC1-DC3, DC2-DC3
        
        # self.min_energy_consumption = 10e9
        # self.max_energy_consumption = 0
        # self.mean_energy_consumption = 715
        # self.std_energy_consumption = 200
        self.energy_stats = []

    def reset(self, seed=0, options=None):
        
        # Set seed if we are not in rllib
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            # tf1.random.set_random_seed(0)

        self.low_level_observations = {}
        self.low_level_infos = {}
        self.heir_obs = {}

        # Reset environments and store initial observations and infos
        for env_id, env in self.datacenters.items():
            obs, info, _ = env.reset()
            self.low_level_observations[env_id] = obs
            self.low_level_infos[env_id] = info
            
            self.heir_obs[env_id] = self.get_dc_variables(env_id)
        
        self.start_index_manager = env.workload_m.time_step
        self.simulated_days = env.days_per_episode
        self.total_computed_workload = 0

        # Initialize metrics
        self.metrics = {
            env_id: {
                'bat_CO2_footprint': [],
                'bat_total_energy_with_battery_KWh': [],
                'ls_tasks_dropped': [],
                'dc_water_usage': [],
                'workload': [],
                'reward': []
            }
            for env_id in self.datacenters
        }

        self.all_done = {env_id: False for env_id in self.datacenters}
        
        return self.flatten_observation(self.heir_obs), self.low_level_infos
    
    def transform_actions(self, actions):
        # Interpret actions
        # actions[0] -> transfer between DC1 and DC2
        # actions[1] -> transfer between DC1 and DC3
        # actions[2] -> transfer between DC2 and DC3

        # Calculate workload to transfer based on action magnitude and direction
        transfer_DC1_DC2 = actions[0]
        transfer_DC1_DC3 = actions[1]
        transfer_DC2_DC3 = actions[2]
        
        # The transfer is a dictionary of dictionaries with the following structure:
        # 'transfer_0': {'receiver': 1/0, 'sender': 0/1, 'workload_to_move': array([actions[0]])
        # 'transfer_1': {'receiver': 2/0, 'sender': 0/2, 'workload_to_move': array([actions[1]])}
        # 'transfer_2': {'receiver': 2/1, 'sender': 1/2, 'workload_to_move': array([actions[2]])}
        # The direction of the transfer is determined by the sign of the action
        
        # Transfer between DC1 and DC2
        transfer_0 = {
            'receiver': 1 if transfer_DC1_DC2 > 0 else 0,
            'sender': 0 if transfer_DC1_DC2 > 0 else 1,
            'workload_to_move': np.array([abs(transfer_DC1_DC2)])
        }
        
        # Transfer between DC1 and DC3
        transfer_1 = {
            'receiver': 2 if transfer_DC1_DC3 > 0 else 0,
            'sender': 0 if transfer_DC1_DC3 > 0 else 2,
            'workload_to_move': np.array([abs(transfer_DC1_DC3)])
        }
        
        # Transfer between DC2 and DC3
        transfer_2 = {
            'receiver': 2 if transfer_DC2_DC3 > 0 else 1,
            'sender': 1 if transfer_DC2_DC3 > 0 else 2,
            'workload_to_move': np.array([abs(transfer_DC2_DC3)])
        }
            
        transfers = {
            'transfer_0': transfer_0,
            'transfer_1': transfer_1,
            'transfer_2': transfer_2
        }
        return transfers



    def step(self, actions):
        actions = self.transform_actions(actions)
        # Move workload between DCs
        
        # TODO: Add the carbon intensity in the action to sort the action in funtion of the reducction in the CI
        self.overassigned_workload = self.safety_enforcement(actions)

        # Step through the low-level agents in each DC
        done = self.low_level_step()

        # Get observations for the next step
        if not done:
            self.heir_obs = {}
            for env_id in self.datacenters:
                self.heir_obs[env_id] = self.get_dc_variables(env_id)

        return self.flatten_observation(self.heir_obs), self.calc_reward(), False, done, {}

    def low_level_step(self, actions: dict = {}):
        
        # Since the top-level agent can change the current workload, we update the observation
        # for the low-level agents here
        for datacenter_id in self.datacenters:
            curr_workload = self.datacenters[datacenter_id].workload_m.get_current_workload()
            # print(f'Current workload for {datacenter_id}: {curr_workload}')
            # On agent_ls, the workload is the 5th element of the array (sine/cos hour day, workload, queue, etc)
            self.low_level_observations[datacenter_id]['agent_ls'][4] = curr_workload
        
        # Compute actions for each dc_id in each environment
        low_level_actions = {}
        for env_id, env_obs in self.low_level_observations.items():
            if self.all_done[env_id]:
                continue
            low_level_actions[env_id] = self.lower_level_actor.compute_actions(env_obs)

            # Override computed low-level actions with provided actions
            low_level_actions[env_id].update(actions.get(env_id, {}))

        # Step through each environment with computed low_level_actions
        self.low_level_infos = {}
        self.low_level_rewards = {}
        for env_id in self.datacenters:
            if self.all_done[env_id]:
                continue

            new_obs, rewards, terminated, truncated, info = self.datacenters[env_id].step(low_level_actions[env_id])
            self.low_level_observations[env_id] = new_obs
            self.all_done[env_id] = terminated['__all__'] or truncated['__all__']

            self.low_level_infos[env_id] = info
            self.low_level_rewards[env_id] = rewards

            # Update metrics for each environment
            env_metrics = self.metrics[env_id]
            env_metrics['bat_CO2_footprint'].append(info['agent_bat']['bat_CO2_footprint'])
            env_metrics['bat_total_energy_with_battery_KWh'].append(info['agent_bat']['bat_total_energy_with_battery_KWh'])
            env_metrics['ls_tasks_dropped'].append(info['agent_ls']['ls_tasks_dropped'])
            env_metrics['dc_water_usage'].append(info['agent_dc']['dc_water_usage'])
            env_metrics['workload'].append(info['agent_ls']['ls_shifted_workload'])

        done = any(self.all_done.values())
        return done

    def flatten_observation(self, observation: dict) -> np.ndarray:
        """
        Flattens the observation dictionary into a plain array, 
        ensuring a consistent order of datacenters and their variables.
        """
        self._original_observation = observation  # Save the original observation

        flattened_obs = []

        for dc_id in sorted(self.datacenters.keys()):  # Ensure consistent order
            dc_obs = observation[dc_id]
            for key in sorted(dc_obs.keys()):  # Ensure consistent order of variables
                flattened_obs.extend(dc_obs[key])

        return np.array(flattened_obs)
    
    def get_dc_variables(self, dc_id: str) -> np.ndarray:
        dc = self.datacenters[dc_id]

        # TODO: check if the variables are normalized with the same values or with min_max values
        obs = {
            'dc_capacity': dc.datacenter_capacity_mw,
            'curr_workload': dc.workload_m.get_current_workload(),
            'weather': dc.weather_m.get_current_weather(),
            'total_power_kw': self.low_level_infos[dc_id]['agent_dc'].get('dc_total_power_kW', 0),
            'ci': dc.ci_m.get_current_ci(),
        }
        
        obs = {key: np.asarray([val]) for (key, val) in obs.items() if key in self.observations}
        
        return obs

    def get_original_observation(self) -> dict:
        """
        Returns the original (unflattened) observation dictionary.
        """
        return self._original_observation

        
    def workload_mapper(self, origin_dc, target_dc, action):
        """
        Translates the workload values from origin dc scale to target dc scale
        """
        assert (action >= 0) & (action <= 1), "action should be a positive fraction"
        return action * (origin_dc.datacenter_capacity_mw / target_dc.datacenter_capacity_mw)

    def _transform_action_array_to_dict(self, action_array):
        # Assuming the array has the following structure:
        # [sender (discrete), receiver (discrete), workload_to_move (continuous)]
        
        # Extract sender, receiver, and workload_to_move from the array
        sender = int(np.round(action_array[0]))
        receiver = int(np.round(action_array[1]))
        workload_to_move = action_array[2]

        # Convert to expected dictionary format
        action_dict = {
            'high_level_policy': {
                'sender': sender,
                'receiver': receiver,
                'workload_to_move': np.array([workload_to_move])
            }
        }

        return action_dict

    def safety_enforcement(self, actions: dict):
        
        # Check if the action is an array instead of a dictionary
        if isinstance(actions, np.ndarray):
        # Transform the array into the expected dictionary format
            actions = self._transform_action_array_to_dict(actions)
        
        # Sort dictionary by workload_to_move
        actions = dict(
            sorted(actions.items(), key=lambda x: x[1]['workload_to_move'], reverse=True))

        # base_workload_on_next_step for all dcs
        self.base_workload_on_curr_step = {dc : self.datacenters[dc].workload_m.get_n_step_future_workload(n=0) for dc in self.datacenters}
        self.base_workload_on_next_step = {dc : self.datacenters[dc].workload_m.get_n_step_future_workload(n=1) for dc in self.datacenters}
        
        # for _, base_workload in self.base_workload_on_next_step.items():
        #     assert (base_workload >= 0) & (base_workload <= 1), "base_workload next_step should be positive and a fraction"
        # for _, base_workload in self.base_workload_on_curr_step.items():
        #     assert (base_workload >= 0) & (base_workload <= 1), "base_workload curr_step should be positive and a fraction"

        overassigned_workload = []
        for _, action in actions.items():
            sender = self.datacenter_ids[action['sender']]
            receiver = self.datacenter_ids[action['receiver']]
            workload_to_move = action['workload_to_move'][0]

            sender_capacity = self.datacenters[sender].datacenter_capacity_mw
            receiver_capacity = self.datacenters[receiver].datacenter_capacity_mw

            # determine the effective workload to be moved and update 
            workload_to_move_mwh = workload_to_move * self.base_workload_on_curr_step[sender] * sender_capacity
            receiver_available_mwh = (1.0 - self.base_workload_on_next_step[receiver]) * receiver_capacity
            effective_movement_mwh = min(workload_to_move_mwh, receiver_available_mwh)
            
            self.base_workload_on_curr_step[sender] -= effective_movement_mwh / sender_capacity
            self.base_workload_on_next_step[receiver] += effective_movement_mwh / receiver_capacity

            # set hysterisis
            #self.set_hysterisis(effective_movement_mwh, sender, receiver)
            
            # keep track of overassigned workload
            overassigned_workload.append(
                (
                    sender, 
                    receiver,
                    (workload_to_move_mwh - effective_movement_mwh) / receiver_capacity
                )
            )
        
        # update individual datacenters with the base_workload_on_curr_step
        for dc, base_workload in self.base_workload_on_curr_step.items():
            self.datacenters[dc].workload_m.set_n_step_future_workload(n = 0, workload = base_workload)

        # update individual datacenters with the base_workload_on_next_step
        for dc, base_workload in self.base_workload_on_next_step.items():
            self.datacenters[dc].workload_m.set_n_step_future_workload(n = 1, workload = base_workload)
        
        # Keep track of the computed workload
        self.total_computed_workload += sum([workload for workload in self.base_workload_on_curr_step.values()])

        return overassigned_workload
    
    def set_hysterisis(self, mwh_to_move: float, sender: str, receiver: str):
        PENALTY = self.penalty
        
        cost_of_moving_mw = mwh_to_move * PENALTY

        self.datacenters[sender].dc_env.set_workload_hysterisis(cost_of_moving_mw)
        self.datacenters[receiver].dc_env.set_workload_hysterisis(cost_of_moving_mw)

    def calc_reward(self) -> float:
        reward = 0
        for dc in self.low_level_infos:
            carbon_footprint = self.low_level_infos[dc]['agent_bat']['bat_CO2_footprint']
            # self.energy_stats.append(carbon_footprint)
            # normalized_energy_consumption = self.normalize_energy_consumption(energy_consumption)
            standardized_energy_consumption = self.standarize_energy_consumption(carbon_footprint)
            reward += standardized_energy_consumption
        return reward


    def standarize_energy_consumption(self, energy_consumption: float) -> float:
        # Negative values to encourage energy saving
        standard_energy = -1.0 * ((energy_consumption - 215000) / 160000)
        return standard_energy

    def normalize_energy_consumption(self, energy_consumption: float) -> float:
        # if energy_consumption < self.min_energy_consumption:
        #     self.min_energy_consumption = energy_consumption
        #     print(f"New min energy consumption: {self.min_energy_consumption}")
        # if energy_consumption > self.max_energy_consumption:
        #     self.max_energy_consumption = energy_consumption
        #     print(f"New max energy consumption: {self.max_energy_consumption}")
        
        # Check to not divide by zero
        if self.max_energy_consumption == self.min_energy_consumption:
            return 0
        return (energy_consumption - self.min_energy_consumption) / (self.max_energy_consumption - self.min_energy_consumption)
    
    
if __name__ == '__main__':
    env = HeirarchicalDCRL(DEFAULT_CONFIG)
    
    done = False
    obs, info = env.reset(seed=0)
    total_reward = 0

    greedy_optimizer = WorkloadOptimizer(list(env.datacenters.keys()))
    
    agent = 0
    max_iterations = 4*24*30
    
    with tqdm(total=max_iterations) as pbar:
        while not done:
            
            # Random actions
            if agent == 0:
                actions = env.action_space.sample()
            
            # Do nothing
            elif agent == 1:
                actions = {
                    'transfer_1': {
                        'sender': 0,
                        'receiver': 0,
                        'workload_to_move': np.array([0.0])
                        }
                    }

            # One-step greedy
            elif agent == 2:
                ci = [obs[dc]['ci'] for dc in env.datacenters]
                actions = {
                    'transfer_1': {
                        'sender': np.argmax(ci),
                        'receiver': np.argmin(ci),
                        'workload_to_move': np.array([1.])
                        }
                    }

            # Multi-step Greedy
            else:
                actions = greedy_optimizer.compute_actions(obs)
            
            obs, reward, terminated, truncated, info = env.step(actions)
            done = truncated
            total_reward += reward

            # Update the progress bar
            pbar.update(1)

    # After simulation, calculate average metrics for each environment
    average_metrics = {
        env_id: {metric: sum(values) / 1e6 for metric, values in env_metrics.items()}
        for env_id, env_metrics in env.metrics.items()
    }

    # Print average metrics for each environment
    for env_id, env_metrics in average_metrics.items():
        print(f"Average Metrics for {env.datacenters[env_id].location}:")
        for metric, value in env_metrics.items():
            print(f"\t{metric}: {value:,.2f}")
        print()  # Blank line for readability

    # Sum metrics across datacenters
    print("Summed metrics across all DC:")
    total_metrics = {}
    for metric in env_metrics:
        total_metrics[metric] = 0.0
        for env_id in average_metrics:
            total_metrics[metric] += average_metrics[env_id][metric]

        print(f'\t{metric}: {total_metrics[metric]:,.2f}')

    print("Total reward = ", total_reward)
    print("Total computed workload = ", env.total_computed_workload)