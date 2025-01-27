import os
import random
import warnings
import copy

import torch
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete

from envs.dcrl_env_harl_partialobs_sb3 import DCRL
from utils.hierarchical_workload_optimizer import WorkloadOptimizer
from utils.low_level_wrapper import LowLevelActorRLLIB, LowLevelActorHARL
from utils.utils_cf import get_init_day

warnings.filterwarnings(
    action="ignore",
    category=UserWarning
    )

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = {
    # DC1
    'config1' : {
        'location': 'ga',
        'cintensity_file': 'CA_NG_&_avgCI.csv',
        'weather_file': 'USA_NY_New.York-LaGuardia.epw',
        'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv',
        'dc_config_file': 'dc_config_dc1.json',
        'datacenter_capacity_mw' : 1.0,
        'flexible_load': 0.4,
        'timezone_shift': 8,
        'month': 7,
        'days_per_episode': 7,
        'partial_obs': True,
        'nonoverlapping_shared_obs_space': True,
        'debug': False,
        'initialize_queue_at_reset': False,
        'agents': ['agent_ls'],
        'workload_baseline': -0.2,
        'temperature_baseline': 0.0,

        },

    # DC2
    'config2' : {
        'location': 'ny',
        'cintensity_file': 'CA_NG_&_avgCI.csv',
        'weather_file': 'USA_AZ_Phoenix-Sky.Harbor.epw',
        'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv',
        'dc_config_file': 'dc_config_dc1.json',
        'datacenter_capacity_mw' : 1.0,
        'flexible_load': 0.4,
        'timezone_shift': 0,
        'month': 7,
        'days_per_episode': 7,
        'partial_obs': True,
        'nonoverlapping_shared_obs_space': True,
        'debug': False,
        'initialize_queue_at_reset': False,
        'agents': ['agent_ls'],
        'workload_baseline': 0.0,
        'temperature_baseline': 0.0,

        },

    # DC3
    'config3' : {
        'location': 'ca',
        'cintensity_file': 'CA_NG_&_avgCI.csv',
        'weather_file': 'USA_CA_San.Jose-Mineta.epw',
        'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv',
        'dc_config_file': 'dc_config_dc1.json',
        'datacenter_capacity_mw' : 1.0,
        'flexible_load': 0.4,
        'timezone_shift': 16,
        'month': 7,
        'days_per_episode': 7,
        'partial_obs': True,
        'nonoverlapping_shared_obs_space': True,
        'debug': False,
        'initialize_queue_at_reset': False,
        'agents': ['agent_ls'],
        'workload_baseline': 0.2,
        'temperature_baseline': 0.0,

        },
    
    # Number of transfers per step
    'num_transfers': 2,
    
    # Number of datacenters in the cluster
    "num_datacenters": 3,

    # List of active low-level agents
    'active_agents': [''],

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

# List of possible locations
LOCATIONS = ["ca", "wa", "tx", "ny", "az", "il", "ga"]

# Function to update only the location in the config dictionary
def update_config_locations(config):
    # Select unique random locations for each datacenter
    selected_locations = random.sample(LOCATIONS, config["num_datacenters"])
    days_per_episode = random.randint(7, 14)

    for i in range(config["num_datacenters"]):
        dc_id = f"config{i+1}"
        # print(f"Updating location for {dc_id}, from {config[dc_id]['location']} to {selected_locations[i]}")
        # config[dc_id]["location"] = selected_locations[i]
        config[dc_id]["workload_baseline"] = random.uniform(-0.5, 0.5)
        config[dc_id]["temperature_baseline"] = random.uniform(-5, 5)
        # config[dc_id]["days_per_episode"] = days_per_episode
        


class HeirarchicalDCRL(gym.Env):

    def __init__(self, config: dict = DEFAULT_CONFIG, random_locations: bool = False):

        self.config = config
        self.random_locations = random_locations  # New parameter
        self.penalty = 0
        # Init all datacenter environments
        DC1 = DCRL(config['config1'])
        DC2 = DCRL(config['config2'])
        DC3 = DCRL(config['config3'])

        self.datacenters = {
            'DC1': DC1,
            'DC2': DC2,
            'DC3': DC3,
        }

        self.datacenter_ids = sorted(list(self.datacenters.keys()))
        
        # Load trained lower level agent
        self.lower_level_actor = LowLevelActorHARL(
            config['low_level_actor_config'],
            config['active_agents']
            )
        

        # Set max episode steps
        self._max_episode_steps = 4 * 24 * DEFAULT_CONFIG['config1']['days_per_episode']
        self.max_episode_steps  = 4 * 24 * DEFAULT_CONFIG['config1']['days_per_episode']


        # Define observation and action space        
        self.common_observations = [
            'time_of_day_sin',
            'time_of_day_cos',
        ]
        
        self.unique_observations = [
            'curr_workload',
            'ci',
            'curr_temperature',
        ]

        
        # Number of datacenters
        num_dcs = len(self.datacenters)
        
        self.max_util = config.get('max_util', 1.0)

        # Each observation component has a shape of (1,) so we need to multiply by the number of observations and datacenters
        observation_dim = len(self.common_observations) + len(self.unique_observations) * num_dcs

        # Define continuous observation space for the flattened observation array
        self.observation_space = Box(
            low=-10.0, high=10.0, shape=(observation_dim,), dtype=np.float32
        )
        

        # Define continuous action space with three variables for transfers
        self.action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32) # DC1-DC2, DC1-DC3, DC2-DC3
        
        self.base_month  = config['config1']['month']
        self.init_day = get_init_day(self.base_month)
        self.ranges_day = [max(0, self.init_day - 7), min(364, self.init_day + 7)]
        
        self.energy_stats = []
        self.debug = True
        
        self.int_seed = None
    
    def seed(self, seed=None):
        self.int_seed = seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        # tf1.random.set_random_seed(0)

    def reset(self, seed=None, options=None):
        
        # Set seed if we are not in rllib
        if seed is not None:
            self.int_seed = seed
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

        self.low_level_observations = {}
        self.low_level_infos = {}
        self.heir_obs = {}
        self.flat_obs = []
        
        # Randomize locations if the flag is set
        if self.random_locations:
            update_config_locations(self.config)  # Randomize locations
            # Reinitialize datacenters with updated configurations
            self.datacenters = {
                'DC1': DCRL(self.config['config1']),
                'DC2': DCRL(self.config['config2']),
                'DC3': DCRL(self.config['config3']),
            }
            self.datacenter_ids = sorted(list(self.datacenters.keys()))
        

        random_init_day  = random.randint(max(0, self.ranges_day[0]), min(364, self.ranges_day[1])) # self.init_day 
        random_init_hour = random.randint(0, 23)
        
        # print(f'Random init day: {random_init_day}, Random init hour: {random_init_hour}')
        
        # Reset environments and store initial observations and infos
        for env_id, env in self.datacenters.items():
            obs, info = env.reset(seed=self.int_seed, random_init_day=random_init_day, random_init_hour=random_init_hour)
            self.low_level_observations[env_id] = obs
            self.low_level_infos[env_id] = info
            
            self.heir_obs[env_id] = self.get_dc_variables(env_id)
        
        # Get common variables after reset (the time manager has internally the hour variable)
        self.heir_obs['__common__'] = self.get_common_variables() if len(self.common_observations) > 0 else {}

        self.start_index_manager = env.workload_m.time_step
        self.simulated_days = env.days_per_episode
        self.total_computed_workload = 0

        # Initialize metrics
        self.metrics = {
            env_id: {
                'bat_CO2_footprint': [],
                'bat_total_energy_with_battery_KWh': [],
                'ls_tasks_dropped': [],
                'ls_overdue_penalty': [],
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
        transfer_DC1_DC2 = np.clip(actions[0], self.action_space.low[0], self.action_space.high[0])
        transfer_DC1_DC3 = np.clip(actions[1], self.action_space.low[0], self.action_space.high[0])
        transfer_DC2_DC3 = np.clip(actions[2], self.action_space.low[0], self.action_space.high[0])
        
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
        self.original_workload, self.total_transferred = self.safety_enforcement(actions)

        # Step through the low-level agents in each DC
        done = self.low_level_step()

        # Get observations for the next step
        if not done:
            self.heir_obs = {}
            for env_id in self.datacenters:
                self.heir_obs[env_id] = self.get_dc_variables(env_id)

        # Get common variables after reset (the time manager has internally the hour variable)
        self.heir_obs['__common__'] = self.get_common_variables() if len(self.common_observations) > 0 else {}
        
        # If you'd like to log CO2 and water usage each step, store them in info
        info = {}
        
        # Let's accumulate across all datacenters
        total_co2 = 0.0
        total_water = 0.0
        for dc_id in self.low_level_infos:
            dc_info = self.low_level_infos[dc_id]
            carbon_footprint = dc_info['agent_bat']['bat_CO2_footprint']
            water_usage = dc_info['agent_dc']['dc_water_usage']
            total_co2 += carbon_footprint
            total_water += water_usage

        # You can store the aggregated metrics in the info dict
        info["co2_footprint"] = total_co2
        info["water_usage"] = total_water
        
        # Include inside the info dict the self.low_level_infos
        info['low_level_infos'] = self.low_level_infos

        # Return as usual
        return self.flatten_observation(self.heir_obs), self.calc_reward(), False, done, info
    
    def low_level_step(self, actions: dict = {}):
        
        # Since the top-level agent can change the current workload, we update the observation
        # for the low-level agents here
        for datacenter_id in self.datacenters:
            curr_workload = self.datacenters[datacenter_id].workload_m.get_current_workload()
            # print(f'Current workload for {datacenter_id}: {curr_workload}')
            # On agent_ls, the workload is the 5th element of the array (sine/cos hour day, workload, queue, etc)
            self.low_level_observations[datacenter_id][4] = curr_workload
            
            # Update the workload in the environment
            self.datacenters[datacenter_id].ls_env.update_workload(curr_workload)
        
        # I am not sure what this action does because we are overwriting the action of the active agents with the computed actions. 
        # Compute actions for each dc_id in each environment
        low_level_actions = {}
        for env_id, env_obs in self.low_level_observations.items():
            if self.all_done[env_id]:
                continue
            # Only the 'active_agents' are considered, the rest are using the default "do-nothing" action
            # low_level_actions[env_id] = self.lower_level_actor.compute_actions(env_obs)

            # Override computed low-level actions with provided actions
            # low_level_actions[env_id].update(actions.get(env_id, {}))
            low_level_actions[env_id] =  actions.get(env_id, {})
        # Step through each environment with computed low_level_actions
        self.low_level_infos = {}
        self.low_level_rewards = {}
        for env_id in self.datacenters:
            if self.all_done[env_id]:
                continue
            
            new_obs, rewards, terminated, truncated, info = self.datacenters[env_id].step(low_level_actions[env_id])
            self.low_level_observations[env_id] = new_obs
            self.all_done[env_id] = terminated or truncated

            self.low_level_infos[env_id] = info
            self.low_level_rewards[env_id] = rewards

            # Update metrics for each environment
            self.metrics[env_id]['bat_CO2_footprint'].append(info['agent_bat']['bat_CO2_footprint'])
            self.metrics[env_id]['bat_total_energy_with_battery_KWh'].append(info['agent_bat']['bat_total_energy_with_battery_KWh'])
            self.metrics[env_id]['ls_tasks_dropped'].append(info['agent_ls']['ls_tasks_dropped'])
            self.metrics[env_id]['dc_water_usage'].append(info['agent_dc']['dc_water_usage'])
            self.metrics[env_id]['workload'].append(info['agent_ls']['ls_shifted_workload'])
            self.metrics[env_id]['ls_overdue_penalty'].append(info['agent_ls']['ls_overdue_penalty'])

            self.heir_obs[env_id] = self.get_dc_variables(env_id)
            
        # Get common variables after reset (the time manager has internally the hour variable)
        self.heir_obs['__common__'] = self.get_common_variables()
        
        done = any(self.all_done.values())
        return done

    def flatten_observation(self, observation: dict) -> np.ndarray:
        """
        Flattens the observation dictionary into a plain array, 
        ensuring a consistent order of datacenters and their variables.
        Handles both single values and arrays.
        """
        self._original_observation = observation  # Save the original observation
        # print(f'Original observation in the main: {self._original_observation}')

        flattened_obs = []

        # First add the common variables
        for key in self.heir_obs['__common__']:
            value = self.heir_obs['__common__'][key]
            if np.isscalar(value):
                flattened_obs.append(value)
            else:
                flattened_obs.extend(np.asarray(value).flatten())
                
        # Then add the variables for each datacenter
        for dc_id in sorted(self.datacenters.keys()):  # Ensure consistent order
            dc_obs = observation[dc_id]
            for key in sorted(dc_obs.keys()):  # Ensure consistent order of variables
                value = dc_obs[key]
                if np.isscalar(value):  # Check if it's a scalar
                    flattened_obs.append(value)
                else:
                    flattened_obs.extend(np.asarray(value).flatten())  # Convert to array and flatten

        self.flat_obs = np.array(flattened_obs, dtype=np.float32)
        
        # print(f'Flattened observation: {self._original_observation}')
        
        return self.flat_obs
    
    def get_common_variables(self):
        """
        Returns the common variables for all datacenters.
        """
        time_of_day = self.datacenters['DC1'].t_m.get_time_of_day()
        return {
            'time_of_day_sin': time_of_day[0],
            'time_of_day_cos': time_of_day[1]
        }
        
    def get_dc_variables(self, dc_id: str) -> np.ndarray:
        dc = self.datacenters[dc_id]

        available_capacity = dc.datacenter_capacity_mw - dc.workload_m.get_current_workload()
        # normalized_ocupacity_last_period = dc.ls_info['ls_previous_computed_workload'] / dc.datacenter_capacity_mw

        # TODO: check if the variables are normalized with the same values or with min_max values
        obs = {
            'dc_capacity': dc.datacenter_capacity_mw,
            'curr_workload': dc.workload_m.get_current_workload(),
            'curr_temperature': dc.weather_m.get_current_weather(),
            'total_power_kw': self.low_level_infos[dc_id]['agent_dc'].get('dc_total_power_kW', 0),
            'ci': dc.ci_m.get_current_ci(),
            'predicted_workload': dc.workload_m.get_forecast_workload(),
            'predicted_weather': dc.weather_m.get_forecast_weather(steps=1),
            'predicted_ci': dc.ci_m.get_forecast_ci()[0],
            'available_capacity': available_capacity,
            'ctafr': dc.dc_env.dc.DC_ITModel_config.CT_REFRENCE_AIR_FLOW_RATE,
            'ct_rated_load': dc.dc_env.dc.DC_ITModel_config.CT_FAN_REF_P/1e6,
        }

        obs = {key: np.asarray([val]) for (key, val) in obs.items() if key in self.unique_observations}

        return obs

    def get_original_observation(self) -> dict:
        """
        Returns the original (unflattened) observation dictionary.
        """
        # print(f'Original observation: {self._original_observation}')
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
        """
        Enforces safety constraints so that after each transfer, no datacenter
        exceeds workload=1.0. This handles multiple incoming transfers
        to the same destination in a single step by applying them sequentially
        in descending order of intended transfer.
        """
        # Check if the action is an array instead of a dictionary
        if isinstance(actions, np.ndarray):
            # Transform the array into the expected dictionary format
            actions = self._transform_action_array_to_dict(actions)
        
        # Sort dictionary by workload_to_move (descending), so largest transfers apply first
        actions = dict(
            sorted(actions.items(), key=lambda x: x[1]['workload_to_move'], reverse=True)
        )

        # Record each datacenter's workload before any changes
        original_workload = {
            dc: self.datacenters[dc].workload_m.get_current_workload() 
            for dc in self.datacenters
        }

        # Optionally store these if your environment logic uses them
        self.base_workload_on_curr_step = {
            dc: original_workload[dc] 
            for dc in self.datacenters
        }
        self.base_workload_on_next_step = {
            dc: self.datacenters[dc].workload_m.get_forecast_workload() 
            for dc in self.datacenters
        }

        # Keep a running snapshot of each datacenter's workload
        current_datacenter_workload = copy.deepcopy(original_workload)

        # net_transfer dict: track how much each DC's final workload changes vs. original
        net_transfer = {dc: 0.0 for dc in self.datacenters}

        total_transferred = 0.0

        # 1) Apply each transfer in order
        for _, action in actions.items():
            sender = self.datacenter_ids[action['sender']]
            receiver = self.datacenter_ids[action['receiver']]
            fraction = action['workload_to_move'][0]  # fraction of sender's current workload

            sender_capacity = self.datacenters[sender].datacenter_capacity_mw
            receiver_capacity = self.datacenters[receiver].datacenter_capacity_mw

            # 2) Compute how many MWh we *intend* to move based on fraction * sender's workload
            #    and check the updated sender's workload from prior transfers
            sender_workload = current_datacenter_workload[sender]
            workload_to_move_mwh = fraction * sender_workload * sender_capacity

            # 3) Check how many MWh the receiver can accept to avoid going beyond self.max_util
            receiver_workload = current_datacenter_workload[receiver]
            receiver_available_mwh = (self.max_util - receiver_workload) * receiver_capacity

            # 4) The effective movement is the min of the intended move and the actual available capacity
            effective_movement_mwh = min(workload_to_move_mwh, receiver_available_mwh)

            # 5) If the movement is positive, apply it
            if effective_movement_mwh > 1e-12:
                # Track how much we actually move
                total_transferred += abs(effective_movement_mwh)

                # Update net transfer for sender and receiver
                net_transfer[sender] -= (effective_movement_mwh / sender_capacity)
                net_transfer[receiver] += (effective_movement_mwh / receiver_capacity)

                # Update the running snapshot so subsequent transfers see the new workloads
                current_datacenter_workload[sender] -= (effective_movement_mwh / sender_capacity)
                current_datacenter_workload[receiver] += (effective_movement_mwh / receiver_capacity)

        # 6) After applying all transfers sequentially, compute final workloads and verify
        for dc, transfer_amount in net_transfer.items():
            new_workload = round(original_workload[dc] + transfer_amount, 6)
            if new_workload < 0 or new_workload > 1:
                raise ValueError(f"Workload for {dc} should be between 0 and 1 after transfer, got {new_workload:.3f}")

            self.datacenters[dc].workload_m.set_current_workload(new_workload)

        # Print the original workload and the final workload assignment and the actions for each datacenter
        # for dc in self.datacenters:
        #     print(f"{dc}: Original={original_workload[dc]:.3f}, Final={self.datacenters[dc].workload_m.get_current_workload():.3f}")
        # for action_id, action in actions.items():
        #     print(f"{action_id}: {action}")

        # 7) Keep track of total computed workload
        self.total_computed_workload += sum(self.base_workload_on_curr_step.values())

        return original_workload, total_transferred


    def safety_enforcement_with_delay(self, actions: dict):
        """
        Enforce safety constraints by transferring workloads between datacenters.
        Supports multiple transfers from a single sender to different receivers.
        Transfers affect the sender's current workload and the receiver's workload in the next timestep.

        Args:
            actions (dict): Dictionary containing transfer actions.

        Returns:
            original_workload (dict): Workload before transfers.
            overassigned_workload (list): List to track any overassigned workloads (currently unused).
        """
        # Sort actions by workload_to_move in descending order (optional for prioritization)
        actions = dict(
            sorted(actions.items(), key=lambda x: x[1]['workload_to_move'], reverse=True))
        
        # Record the original workloads before any transfers
        original_workload = {
            dc: self.datacenters[dc].workload_m.get_current_workload() for dc in self.datacenters
        }

        # Initialize dictionaries to track net transfers and future transfers
        net_transfer = {dc: 0.0 for dc in self.datacenters}          # For senders
        future_transfers = {dc: 0.0 for dc in self.datacenters}     # For receivers

        # Initialize a dictionary to track remaining transferable workload per sender
        remaining_transfer = {
            dc: original_workload[dc] * self.datacenters[dc].datacenter_capacity_mw  # Assuming workload is normalized
            for dc in self.datacenters
        }

        # Process each transfer action
        for transfer_id, action in actions.items():
            sender_idx = action['sender']
            receiver_idx = action['receiver']
            workload_fraction = action['workload_to_move'][0]  # Fraction between 0 and 1

            sender = self.datacenter_ids[sender_idx]
            receiver = self.datacenter_ids[receiver_idx]

            # Skip self-transfers
            if sender == receiver:
                if self.debug:
                    print(f"[{transfer_id}] Skipping self-transfer for {sender}.")
                continue

            # Calculate the workload to move in MWh based on sender's capacity
            sender_capacity_mw = self.datacenters[sender].datacenter_capacity_mw
            workload_to_move_mwh = workload_fraction * original_workload[sender] * sender_capacity_mw

            if self.debug:
                print(f"[{transfer_id}] Attempting to transfer {workload_fraction:.3f} "
                    f"({workload_to_move_mwh:.3f} MWh) from {sender} to {receiver}.")

            # Check sender's remaining transferable workload
            if workload_to_move_mwh > remaining_transfer[sender]:
                # Adjust the workload to the remaining transferable amount
                workload_to_move_mwh = remaining_transfer[sender]
                workload_fraction = workload_to_move_mwh / (original_workload[sender] * sender_capacity_mw)
                if self.debug:
                    print(f"[{transfer_id}] Adjusted transfer to {workload_fraction:.3f} "
                        f"({workload_to_move_mwh:.3f} MWh) due to sender's remaining capacity.")
            
            if workload_to_move_mwh <= 0:
                if self.debug:
                    print(f"[{transfer_id}] No transferable workload remaining for {sender}. Skipping transfer to {receiver}.")
                continue  # Nothing to transfer

            # Calculate available capacity in the receiver for the next timestep
            receiver_capacity_mw = self.datacenters[receiver].datacenter_capacity_mw
            receiver_forecast_workload = self.datacenters[receiver].workload_m.get_forecast_workload()
            receiver_available_mwh = (self.max_util - receiver_forecast_workload) * receiver_capacity_mw

            # Subtract any already scheduled transfers to this receiver
            receiver_available_mwh -= future_transfers[receiver] * receiver_capacity_mw

            if self.debug:
                print(f"[{transfer_id}] {receiver} has {receiver_available_mwh:.3f} MWh available in next timestep.")

            # Determine the effective movement (cannot exceed receiver's available capacity)
            effective_movement_mwh = min(workload_to_move_mwh, receiver_available_mwh)
            effective_transfer_fraction = effective_movement_mwh / receiver_capacity_mw  # Convert back to normalized workload

            if effective_transfer_fraction <= 0:
                if self.debug:
                    print(f"[{transfer_id}] No available capacity in {receiver} for transfer from {sender}. Skipping.")
                continue  # Skip this transfer as there's no available capacity

            # Update net transfer for the sender (workload is leaving)
            transfer_fraction = effective_transfer_fraction / original_workload[sender] if original_workload[sender] > 0 else 0
            net_transfer[sender] -= transfer_fraction  # Outgoing workload (normalized)

            # Accumulate transfer for the receiver's future workload
            future_transfers[receiver] += effective_transfer_fraction

            # Update the remaining transferable workload for the sender
            remaining_transfer[sender] -= effective_movement_mwh

            if self.debug:
                print(f"[{transfer_id}] Transferring {effective_transfer_fraction:.3f} fraction "
                    f"({effective_movement_mwh:.3f} MWh) from {sender} to {receiver}.")

        # Apply net transfers to senders' current workloads
        for sender, transfer in net_transfer.items():
            new_workload = round(original_workload[sender] + transfer, 6)
            new_workload = np.clip(new_workload, 0.0, 1.0)
            self.datacenters[sender].workload_m.set_current_workload(new_workload)
            if self.debug:
                print(f"[Update] Sender {sender}: Workload updated from {original_workload[sender]:.3f} "
                    f"to {new_workload:.3f}.")

        # Apply future transfers to receivers' future workloads
        for receiver, transfer in future_transfers.items():
            if transfer > 0:
                current_forecast = self.datacenters[receiver].workload_m.get_forecast_workload()
                new_future_workload = current_forecast + transfer
                new_future_workload = min(new_future_workload, 1.0)  # Ensure it doesn't exceed max utility

                # Set the future workload using Workload_Manager
                self.datacenters[receiver].workload_m.set_future_workload(new_future_workload)

                if self.debug:
                    print(f"[Update] Receiver {receiver}: Future workload updated from {current_forecast:.3f} "
                        f"to {new_future_workload:.3f}.")

        # Currently, overassigned_workload is unused. You can implement tracking if needed.
        overassigned_workload = []

        return original_workload, overassigned_workload
    
    def safety_enforcement_with_delay_v2(self, actions: dict):
        """
        Enforce safety constraints by transferring workloads between datacenters.
        Supports multiple transfers from a single sender to different receivers.
        Transfers affect the sender's current workload and the receiver's workload in the next timestep.
        
        Args:
            actions (dict): Dictionary containing transfer actions.
        
        Returns:
            original_workload (dict): Workload before transfers.
            overassigned_workload (list): List to track any overassigned workloads (currently unused).
        """
        # Sort actions by workload_to_move in descending order (optional for prioritization)
        actions = dict(
            sorted(actions.items(), key=lambda x: x[1]['workload_to_move'], reverse=True))
        
        # Record the original workloads before any transfers
        original_workload = {
            dc: self.datacenters[dc].workload_m.get_current_workload() for dc in self.datacenters
        }

        if self.debug:
            print("\n--- Safety Enforcement Start ---")
            print("Original Workloads:", original_workload)

        # Initialize dictionaries to track net transfers and future transfers
        net_transfer = {dc: 0.0 for dc in self.datacenters}          # For senders
        future_transfers = {dc: 0.0 for dc in self.datacenters}     # For receivers

        # Initialize a dictionary to track cumulative transfer fractions per sender
        cumulative_transfer_fraction = {dc: 0.0 for dc in self.datacenters}

        # Process each transfer action
        for transfer_id, action in actions.items():
            sender_idx = action['sender']
            receiver_idx = action['receiver']
            workload_fraction = action['workload_to_move'][0]  # Fraction between 0 and 1

            sender = self.datacenter_ids[sender_idx]
            receiver = self.datacenter_ids[receiver_idx]

            # Skip self-transfers
            if sender == receiver:
                if self.debug:
                    print(f"[{transfer_id}] Skipping self-transfer for {sender}.")
                continue

            # Calculate the intended transfer fraction based on original workload
            # Ensure that cumulative_transfer_fraction[sender] + workload_fraction <=1
            available_fraction = 1.0 - cumulative_transfer_fraction[sender]
            if available_fraction <= 0:
                if self.debug:
                    print(f"[{transfer_id}] Sender {sender} has no remaining transferable workload. Skipping.")
                continue

            # Adjust the workload fraction if it exceeds available_fraction
            if workload_fraction > available_fraction:
                adjusted_fraction = available_fraction
                if self.debug:
                    print(f"[{transfer_id}] Adjusting workload_fraction from {workload_fraction:.3f} to {adjusted_fraction:.3f} due to sender's remaining capacity.")
                workload_fraction = adjusted_fraction

            # Update cumulative_transfer_fraction
            cumulative_transfer_fraction[sender] += workload_fraction

            # Calculate the workload to move in MWh based on sender's original workload
            sender_capacity_mw = self.datacenters[sender].datacenter_capacity_mw
            workload_to_move_mwh = workload_fraction * original_workload[sender] * sender_capacity_mw

            if self.debug:
                print(f"[{transfer_id}] Attempting to transfer {workload_fraction:.3f} "
                    f"({workload_to_move_mwh:.3f} MWh) from {sender} to {receiver}.")

            # Calculate available capacity in the receiver for the next timestep
            receiver_capacity_mw = self.datacenters[receiver].datacenter_capacity_mw
            receiver_forecast_workload = self.datacenters[receiver].workload_m.get_forecast_workload()
            receiver_available_mwh = (self.max_util - receiver_forecast_workload) * receiver_capacity_mw

            # Subtract any already scheduled transfers to this receiver
            receiver_available_mwh -= future_transfers[receiver] * receiver_capacity_mw

            if self.debug:
                print(f"[{transfer_id}] {receiver} has {receiver_available_mwh:.3f} MWh available in next timestep.")

            # Determine the effective movement (cannot exceed receiver's available capacity)
            effective_movement_mwh = min(workload_to_move_mwh, receiver_available_mwh)
            effective_transfer_fraction = effective_movement_mwh / (original_workload[sender] * sender_capacity_mw) if (original_workload[sender] * sender_capacity_mw) > 0 else 0.0  # Fraction of sender's workload

            if self.debug:
                if effective_transfer_fraction < workload_fraction:
                    print(f"[{transfer_id}] Adjusted transfer fraction to {effective_transfer_fraction:.3f} "
                        f"due to receiver's available capacity.")

            # If effective_transfer_fraction is less than desired, adjust the cumulative_transfer_fraction
            if effective_transfer_fraction < workload_fraction:
                # Reduce the cumulative_transfer_fraction accordingly
                cumulative_transfer_fraction[sender] -= (workload_fraction - effective_transfer_fraction)
                workload_fraction = effective_transfer_fraction
                workload_to_move_mwh = workload_fraction * original_workload[sender] * sender_capacity_mw
                if self.debug:
                    print(f"[{transfer_id}] Final transfer: {workload_fraction:.3f} fraction "
                        f"({workload_to_move_mwh:.3f} MWh).")

            if effective_transfer_fraction <= 0:
                if self.debug:
                    print(f"[{transfer_id}] No available capacity in {receiver} for transfer from {sender}. Skipping.")
                continue  # Nothing to transfer

            # Update net transfer for the sender (workload is leaving)
            transfer_fraction = workload_fraction
            net_transfer[sender] += transfer_fraction  # Outgoing workload (normalized)

            # Accumulate transfer for the receiver's future workload
            future_transfers[receiver] += workload_to_move_mwh / receiver_capacity_mw

            if self.debug:
                print(f"[{transfer_id}] Transferring {workload_fraction:.3f} fraction "
                    f"({workload_to_move_mwh:.3f} MWh) from {sender} to {receiver}.")

        # Apply net transfers to senders' current workloads
        for sender, transfer_fraction in net_transfer.items():
            # sender's workload after transfer
            new_workload = original_workload[sender] - transfer_fraction * original_workload[sender]
            new_workload = np.clip(new_workload, 0.0, 1.0)
            self.datacenters[sender].workload_m.set_current_workload(new_workload)
            if self.debug:
                print(f"[Update] Sender {sender}: Workload updated from {original_workload[sender]:.3f} "
                    f"to {new_workload:.3f}.")

        # Apply future transfers to receivers' future workloads
        for receiver, transfer_fraction in future_transfers.items():
            if transfer_fraction > 0:
                current_forecast = self.datacenters[receiver].workload_m.get_forecast_workload()
                new_future_workload = current_forecast + transfer_fraction
                new_future_workload = min(new_future_workload, 1.0)  # Ensure it doesn't exceed max utility

                # Set the future workload using Workload_Manager
                self.datacenters[receiver].workload_m.set_future_workload(new_future_workload)

                if self.debug:
                    print(f"[Update] Receiver {receiver}: Future workload updated from {current_forecast:.3f} "
                        f"to {new_future_workload:.3f}.")

        if self.debug:
            print("--- Safety Enforcement End ---\n")

        # Currently, overassigned_workload is unused. You can implement tracking if needed.
        overassigned_workload = []

        return original_workload, overassigned_workload



    def set_hysterisis(self, mwh_to_move: float, sender: str, receiver: str):
        PENALTY = self.penalty
        
        cost_of_moving_mw = mwh_to_move * PENALTY

        self.datacenters[sender].dc_env.set_workload_hysterisis(cost_of_moving_mw)
        self.datacenters[receiver].dc_env.set_workload_hysterisis(cost_of_moving_mw)

    def calc_reward(self) -> float:
        reward = 0
        for dc in self.low_level_infos:
            carbon_footprint = self.low_level_infos[dc]['agent_bat']['bat_CO2_footprint']
            water_usage = self.low_level_infos[dc]['agent_dc']['dc_water_usage']
            
            standarized_carbon_footprint = -1.0 * ((carbon_footprint - 114825) / 99179)
            standarized_water_usage = -1.0 * ((water_usage - 650) / 438)
            
            reward += standarized_carbon_footprint + standarized_water_usage
        
        # Define penalty parameters
        transfer_cost_coeff = 0.01  # Adjust this coefficient based on experimentation
        max_transfer_penalty = 5.0  # Maximum penalty per step

        # Calculate penalty
        transfer_penalty = transfer_cost_coeff * self.total_transferred

        # Optionally, cap the penalty
        transfer_penalty = min(transfer_penalty, max_transfer_penalty)

        # Subtract penalty from reward
        reward -= transfer_penalty
        return reward


    def standarize_carbon_footprint(self, carbon_footprint: float) -> float:
        # Negative values to encourage energy saving
        standard_carbon_footprint = -1.0 * ((carbon_footprint - 95000) / 50000)
        return standard_carbon_footprint

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