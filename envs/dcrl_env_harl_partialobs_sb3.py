import os
import random
from typing import Optional, Tuple, Union

import gymnasium
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from utils import reward_creator
from utils.base_agents import (BaseBatteryAgent, BaseHVACAgent,
                               BaseLoadShiftingAgent)
from utils.make_envs_pyenv import (make_bat_fwd_env, make_dc_pyeplus_env,
                                   make_ls_env)
from utils.managers import (CI_Manager, Time_Manager, Weather_Manager,
                            Workload_Manager)
from utils.utils_cf import get_energy_variables, get_init_day, obtain_paths


class EnvConfig(dict):

    # Default configuration for this environment. New parameters should be
    # added here
    DEFAULT_CONFIG = {
        # Agents active
        'agents': ['agent_ls', 'agent_dc', 'agent_bat'],

        # Datafiles
        'location': 'ny',
        'cintensity_file': 'NYIS_NG_&_avgCI.csv',
        'weather_file': 'USA_NY_New.York-Kennedy.epw',
        'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv',
        
        # Capacity (MW) of the datacenter
        'datacenter_capacity_mw': 1,
        
        # Timezone shift
        'timezone_shift': 0,
        
        # Days per simulated episode
        'days_per_episode': 30,
        
        # Maximum battery capacity
        'max_bat_cap_Mw': 2,
        
        # Data center configuration file
        'dc_config_file': 'dc_config.json',
        
        # weight of the individual reward (1=full individual, 0=full collaborative, default=0.8)
        'individual_reward_weight': 0.8,
        
        # flexible load ratio of the total workload
        'flexible_load': 0.1,
        
        # Specify reward methods. These are defined in utils/reward_creator.
        'ls_reward': 'default_ls_reward',
        'dc_reward': 'default_dc_reward',
        'bat_reward': 'default_bat_reward',

        # Evaluation flag that is required by the load-shifting environment
        # To be set only during offline evaluation
        'evaluation': False,

        # Set this to True if an agent (like MADDPG) returns continuous actions,
        "actions_are_logits": False,
        
        'debug': False,
        
        'initialize_queue_at_reset': False,
    }

    def __init__(self, raw_config):
        dict.__init__(self, self.DEFAULT_CONFIG.copy())

        # Override defaults with the passed config
        for key, val in raw_config.items():
            self[key] = val



class DCRL(gym.Env):
    def __init__(self, env_config):
        '''
        Args:
            env_config (dict): Dictionary containing parameters as defined in 
                            EnvConfig above
        '''
        super().__init__()

        # Initialize the environment config
        env_config = EnvConfig(env_config)

        # create environments and agents
        self.agents = env_config['agents']
        self.location = env_config['location']
        
        self.ci_file = env_config['cintensity_file']
        self.weather_file = env_config['weather_file']
        self.workload_file = env_config['workload_file']
        
        self.max_bat_cap_Mw = env_config['max_bat_cap_Mw']
        self.indv_reward = env_config['individual_reward_weight']
        self.collab_reward = (1 - self.indv_reward) / 2
        
        self.flexible_load = env_config['flexible_load']

        self.datacenter_capacity_mw = env_config['datacenter_capacity_mw']
        self.dc_config_file = env_config['dc_config_file']
        self.timezone_shift = env_config['timezone_shift']
        self.days_per_episode = env_config['days_per_episode'] + np.random.randint(-5, 30)
        self.initialize_queue_at_reset =  env_config.get('initialize_queue_at_reset', False)
        
        # print(f'Simulating {self.days_per_episode} days')
        
        self.workload_baseline = env_config.get('workload_baseline', False)
        self.debug = env_config.get('debug', False)
        
        # # Assign month according to worker index, if available
        # if hasattr(env_config, 'worker_index'):
        #     self.month = int((env_config.worker_index - 1) % 12)
        #     print(f'Changing the simulated month to {self.month}')
        # else:
        self.month = env_config.get('month')
        # print(f'Simulating month {self.month}')

        self.evaluation_mode = env_config['evaluation']

        self._agent_ids = set(self.agents)

        ci_loc, wea_loc = obtain_paths(self.location)
        
        ls_reward_method = 'default_ls_reward' if not 'ls_reward' in env_config.keys() else env_config['ls_reward']
        self.ls_reward_method = reward_creator.get_reward_method(ls_reward_method)

        dc_reward_method =  'default_dc_reward' if not 'dc_reward' in env_config.keys() else env_config['dc_reward']
        self.dc_reward_method = reward_creator.get_reward_method(dc_reward_method)
        
        bat_reward_method = 'default_bat_reward' if not 'bat_reward' in env_config.keys() else env_config['bat_reward']
        self.bat_reward_method = reward_creator.get_reward_method(bat_reward_method)
        
        n_vars_energy, n_vars_battery = 0, 0  # for partial observability (for p.o.)
        n_vars_ci = 32
        self.ls_env = make_ls_env(month=self.month, test_mode=self.evaluation_mode, n_vars_ci=n_vars_ci, flexible_workload_ratio=self.flexible_load,
                                  n_vars_energy=n_vars_energy, n_vars_battery=n_vars_battery, queue_max_len=2000, initialize_queue_at_reset=self.initialize_queue_at_reset)
        self.dc_env, _ = make_dc_pyeplus_env(month=self.month + 1, location=ci_loc, max_bat_cap_Mw=self.max_bat_cap_Mw, use_ls_cpu_load=True, 
                                             datacenter_capacity_mw=self.datacenter_capacity_mw, dc_config_file=self.dc_config_file, add_cpu_usage=False)
        self.bat_env = make_bat_fwd_env(month=self.month, max_bat_cap_Mwh=self.dc_env.ranges['max_battery_energy_Mwh'], 
                                        max_dc_pw_MW=self.dc_env.ranges['Facility Total Electricity Demand Rate(Whole Building)'][1] / 1e6, 
                                        dcload_max=self.dc_env.ranges['Facility Total Electricity Demand Rate(Whole Building)'][1],
                                        dcload_min=self.dc_env.ranges['Facility Total Electricity Demand Rate(Whole Building)'][0],
                                        n_fwd_steps=n_vars_ci)

        self.bat_env.dcload_max = self.dc_env.power_ub_kW / 4 # Assuming 15 minutes timestep. Kwh
        
        self.bat_env.dcload_min = self.dc_env.power_lb_kW / 4 # Assuming 15 minutes timestep. Kwh
        
        self._obs_space_in_preferred_format = True
        
        self.observation_space = []
        self.action_space = []
        
        self.base_agents = {}
        flexible_load = 0
        
        # self.max_value_state = 0
        # self.min_value_state = 0
        
        # Create the observation/action space if the agent is used for training.
        # Otherwise, create the base agent for the environment.
        if "agent_ls" in self.agents:
            self.observation_space = self.ls_env.observation_space
            self.action_space = self.ls_env.action_space
            flexible_load = self.flexible_load
        else:
            self.base_agents["agent_ls"] = BaseLoadShiftingAgent()
            
        if "agent_dc" in self.agents:
            self.observation_space = self.dc_env.observation_space
            self.action_space = self.dc_env.action_space
        else:
            self.base_agents["agent_dc"] = BaseHVACAgent()
            
        if "agent_bat" in self.agents:
            self.observation_space = self.bat_env.observation_space
            self.action_space = self.bat_env.action_space
        else:
            self.base_agents["agent_bat"] = BaseBatteryAgent()
            
        # ls_state[0:10]->10 variables
        # dc_state[4:9] & bat_state[5]->5+1 variables

        # Create the managers: date/hour/time manager, workload manager, weather manager, and CI manager.
        self.init_day = get_init_day(self.month)
        self.ranges_day = [max(0, self.init_day - 7), min(364, self.init_day + 7)]
        self.t_m = Time_Manager(self.init_day, timezone_shift=self.timezone_shift, days_per_episode=self.days_per_episode)
        self.workload_m = Workload_Manager(init_day=self.init_day, workload_filename=self.workload_file, timezone_shift=self.timezone_shift, debug=self.debug, workload_baseline=self.workload_baseline)
        self.weather_m = Weather_Manager(init_day=self.init_day, location=wea_loc, filename=self.weather_file, timezone_shift=self.timezone_shift, debug=self.debug)
        self.ci_m = CI_Manager(init_day=self.init_day, location=ci_loc, filename=self.ci_file, future_steps=n_vars_ci, timezone_shift=self.timezone_shift)

        # This actions_are_logits is True only for MADDPG, because RLLib defines MADDPG only for continuous actions.
        self.actions_are_logits = env_config.get("actions_are_logits", False)
                
    

    def set_seed(self, seed=None):
        """Set the random seed for the environment."""
        seed = seed or 1
        np.random.seed(seed)
        random.seed(seed)
        if torch is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self._seed_spaces(seed)

    def _seed_spaces(self, seed):
        """Seed the action and observation spaces."""
        if hasattr(self, 'action_space') and hasattr(self.action_space, 'seed'):
            self.action_space.seed(seed)
        if hasattr(self, 'observation_space') and hasattr(self.observation_space, 'seed'):
            self.observation_space.seed(seed)

    def extract_ci_features(self, ci_values, current_ci):
        # Calculate statistical measures
        ci_mean = np.mean(ci_values)
        ci_std = np.std(ci_values)
        ci_variance = ci_std ** 2

        # Calculate gradients
        ci_gradient = np.gradient(np.hstack((current_ci, ci_values)))
        # ci_second_derivative = np.gradient(ci_gradient)

        # Normalize gradients
        # ci_gradient_norm = ci_gradient / (np.max(np.abs(ci_gradient)) + 1e-8)
        # ci_second_derivative_norm = ci_second_derivative / (np.max(np.abs(ci_second_derivative)) + 1e-8)

        # Identify peaks and valleys
        peaks = np.where((ci_gradient[:-1] > 0) & (ci_gradient[1:] <= 0))[0]
        valleys = np.where((ci_gradient[:-1] < 0) & (ci_gradient[1:] >= 0))[0]

        # Time to next peak or valley
        time_to_next_peak = peaks[0] if len(peaks) > 0 else len(ci_values)
        time_to_next_valley = valleys[0] if len(valleys) > 0 else len(ci_values)

        # Relative CI position
        ci_percentile = (current_ci - ci_mean) / (ci_std + 1e-8)
        # Limit the ci_percentile
        ci_percentile = np.clip(ci_percentile, -10, 10)

        # Assemble features
        ci_features = np.array([
                                ci_mean,
                                # ci_std,
                                # ci_percentile,
                                time_to_next_peak/len(ci_values),
                                time_to_next_valley/len(ci_values),
                            ])

        return ci_features

    def _create_ls_state(self, t_i, current_workload, queue_status, current_ci, ci_future, ci_past, next_workload, current_out_temperature, next_out_temperature, next_n_out_temperature, oldest_task_age, average_task_age, ls_task_age_histogram, ls_info):
        """
        Create the state of the load shifting environment.

        Returns:
            np.ndarray: State of the load shifting environment.
        """
        hour_sin_cos = t_i[:2]

        # # CI Trend analysis
        trend_smoothing_window = 4
        smoothed_ci_future = np.convolve(np.hstack((current_ci, ci_future[:16])), np.ones(trend_smoothing_window), 'valid') / trend_smoothing_window
        smoothed_ci_past = np.convolve(np.hstack((ci_past, current_ci)), np.ones(trend_smoothing_window), 'valid') / trend_smoothing_window
        
        # # Slope the next 4 hours of CI and the previous 1 hour of CI
        ci_future_slope = np.polyfit(range(len(smoothed_ci_future)), smoothed_ci_future, 1)[0]
        ci_past_slope = np.polyfit(range(len(smoothed_ci_past)), smoothed_ci_past, 1)[0]

        # # Extract features for future and past CI
        ci_future_features = self.extract_ci_features(ci_future, current_ci)
        # # ci_past_features = self.extract_ci_features(ci_past, current_ci)

        # # Assemble CI features
        ci_features = np.hstack([
                        ci_future_slope, ci_past_slope,
                        ci_future_features
                    ])

        # # Weather trend analysis
        # temperature_slope = np.polyfit(range(len(next_n_out_temperature) + 1), np.hstack([current_out_temperature, next_n_out_temperature]), 1)[0]
        
        # # Extract features for the future temperature
        # temperature_features = self.extract_ci_features(next_n_out_temperature, current_out_temperature)
        
        # # Assemble temperature features
        # temperature_features = np.hstack([
        #                                 temperature_slope, temperature_features
        #                             ])
        
        # # Combine all features into the state
        # ls_state = np.float32(np.hstack((
        #                                 hour_sin_cos,
        #                                 current_ci,
        #                                 ci_features,
        #                                 oldest_task_age,
        #                                 average_task_age,
        #                                 queue_status,
        #                                 current_workload,
        #                                 current_out_temperature,
        #                                 temperature_features,
        #                                 ls_task_age_histogram
        #                             )))
        
        dropped_tasks = np.minimum(ls_info['ls_tasks_dropped'], 4) / 4 # Cropping the number of dropped tasks to 4
        overdue_tasks = np.minimum(ls_info['ls_overdue_penalty'], 10) / 10 # Cropping the number of overdue tasks to 10
        
        ls_state = np.float32(np.hstack((
            # hour_sin_cos,
            current_ci,
            ci_features,
            current_workload,
            oldest_task_age,
            # queue_status,
            ls_task_age_histogram,
            dropped_tasks,
            overdue_tasks
            
                                    )))
        # if len(ls_state) != 26:
            # print(f'Error: {len(ls_state)}')
            
        # Raise an error if there is a nan in the state
        if np.isnan(ls_state).any():
            raise ValueError("State contains NaN values")
        return ls_state
    
    def _create_dc_state(self, current_workload, next_workload, current_out_temperature, next_out_temperature):
        """
        Create the state of the data center environment.

        Returns:
            np.ndarray: State of the data center environment.
        """
        # If liquid cooling model:
        if 'dc_coo_mov_flow_actual' in self.dc_info.keys():
            pump_speed = (self.dc_info.get('dc_coo_mov_flow_actual', 0.05) - self.dc_env.min_pump_speed) / (self.dc_env.max_pump_speed-self.dc_env.min_pump_speed)
            supply_temp = (self.dc_info.get('dc_supply_liquid_temp', 27) - self.dc_env.min_supply_temp) / (self.dc_env.max_supply_temp-self.dc_env.min_supply_temp)

            dc_state = np.float32(np.hstack((
                                                current_workload, 
                                                next_workload,
                                                current_out_temperature,
                                                next_out_temperature,
                                                pump_speed,
                                                supply_temp
                                            )))
            
        else:
            air_supply_temp = (self.dc_env.raw_curr_stpt - self.dc_env.min_temp) / (self.dc_env.max_temp-self.dc_env.min_temp)
            dc_state = np.float32(np.hstack((
                                                current_workload, 
                                                next_workload,
                                                current_out_temperature,
                                                next_out_temperature,
                                            )))
        
        return dc_state


    def _create_bat_state(self, current_workload, next_workload, current_c_i, current_temperature):
        """
        Create the state of the battery environment.

        Returns:
            np.ndarray: State of the battery environment.
        """
        bat_state = np.float32(np.hstack((
                                            current_workload,
                                            next_workload,
                                            current_c_i,
                                            current_temperature
                                        )))
        return bat_state


    def reset(self, seed=0, random_init_day=None, random_init_hour=None):
        """
        Reset the environment.

        Args:
            seed (int, optional): Random seed.
            options (dict, optional): Environment options.

        Returns:
            states (dict): Dictionary of states.
            infos (dict): Dictionary of infos.
        """
        
        self.set_seed(seed)
        
        random_init_day  = random.randint(max(0, self.ranges_day[0]), min(364, self.ranges_day[1])) # self.init_day 
        random_init_hour = random.randint(0, 23)
        
        # Reset termination and reward flags for all agents
        self.ls_terminated = self.dc_terminated = self.bat_terminated = False
        self.ls_truncated = self.dc_truncated = self.bat_truncated = False
        self.ls_reward = self.dc_reward = self.bat_reward = 0

        # Reset the managers
        random_init_hour = random_init_hour + self.timezone_shift
        
        # Fix the random initialization of the day and hour if random_init_day > 23
        if random_init_hour > 23:
            random_init_hour = random_init_hour - 24
            random_init_day += 1
            if random_init_day > 364:
                random_init_day = 0
        
        # print(f'Reseting the environment {self.location.upper()} with day {random_init_day} and hour {random_init_hour}')
        self.current_hour = random_init_hour
        
        t_i = self.t_m.reset(init_day=random_init_day, init_hour=random_init_hour)
        workload = self.workload_m.reset(init_day=random_init_day, init_hour=random_init_hour)  # Workload manager
        temp, norm_temp, wet_bulb, norm_wet_bulb = self.weather_m.reset(init_day=random_init_day, init_hour=random_init_hour)  # Weather manager
        ci_i, ci_i_future, ci_i_denorm = self.ci_m.reset(init_day=random_init_day, init_hour=random_init_hour)  # CI manager. ci_i -> CI in the current timestep.
        
        # Set the external ambient temperature to data center environment
        self.dc_env.set_ambient_temp(temp, wet_bulb)
        
        # Update the workload of the load shifting environment
        self.ls_env.update_workload(workload)
        self.ls_env.update_current_date(random_init_day, random_init_hour)
        
        # Reset all the environments
        ls_s, self.ls_info = self.ls_env.reset()
        self.dc_state, self.dc_info = self.dc_env.reset()
        bat_s, self.bat_info = self.bat_env.reset()
                
        current_workload = self.workload_m.get_current_workload()
        next_workload = self.workload_m.get_next_workload()
        
        current_out_temperature = self.weather_m.get_current_temperature()
        next_out_temperature = self.weather_m.get_next_temperature()
        next_n_out_temperature = self.weather_m.get_n_next_temperature(n=32)
        
        # ls_state -> [time (sine/cosine enconded), original ls observation, current+future normalized CI]
        queue_status = self.ls_info['ls_norm_tasks_in_queue']
        ci_i_past = self.ci_m.get_n_past_ci(n=16)
        oldest_task_age = self.ls_info['ls_oldest_task_age']
        average_task_age = self.ls_info['ls_average_task_age']
        ls_task_age_histogram = self.ls_info['ls_task_age_histogram']
        self.ls_state = self._create_ls_state(t_i, workload, queue_status, ci_i, ci_i_future, ci_i_past, next_workload, current_out_temperature, next_out_temperature, next_n_out_temperature, oldest_task_age, average_task_age, ls_task_age_histogram, self.ls_info)
        
        self.dc_state = self._create_dc_state(current_workload, next_workload, current_out_temperature, next_out_temperature)
        
        # bat_state -> [time (sine/cosine enconded), battery SoC, current+future normalized CI]
        # self.bat_state = np.float32(np.hstack((t_i, bat_s, ci_i_future)))
        self.bat_state = self._create_bat_state(current_workload, next_workload, ci_i, temp)
        
        # Update ci in the battery environment
        self.bat_env.update_ci(ci_i_denorm, ci_i_future[0])

        # States should be a dictionary with agent names as keys and their observations as values
        self.infos = {}
        # Update states and infos considering the agents defined in the environment config self.agents.
        if "agent_ls" in self.agents:
            states = self.ls_state
        if "agent_dc" in self.agents:
            states = self.dc_state
        if "agent_bat" in self.agents:
            states = self.bat_state

        # Prepare the infos dictionary with common and individual agent information
        self.infos = {
            'agent_ls': self.ls_info,
            'agent_dc': self.dc_info,
            'agent_bat': self.bat_info,
            '__common__': {
                'time': t_i,
                'workload': workload,
                'weather': temp,
                'ci': ci_i,
                'ci_future': ci_i_future,
                'states': {
                    'agent_ls': self.ls_state,
                    'agent_dc': self.dc_state,
                    'agent_bat': self.bat_state
                }
            }
        }
        available_actions = None
                
        return states, self.infos 

            
    
    def step(self, action_dict):
        """
        Step the environment.

        Args:
            action_dict: Dictionary of actions of each agent defined in self.agents.
  
        Returns:
            obs (dict): Dictionary of observations/states.
            rews (dict): Dictionary of rewards.
            terminated (dict): Dictionary of terminated flags.
            truncated (dict): Dictionary of truncated flags.
            infos (dict): Dictionary of infos.
        """
        obs, rew, terminateds, truncateds, info = {}, {}, {}, {}, {}
        terminateds["__all__"] = False
        truncateds["__all__"] = False
        
        # Perform actions for each agent and update their respective environments
        self._perform_actions(action_dict)
    
        # Step the managers (time, workload, weather, CI) (t+1)
        day, hour, t_i, terminal = self.t_m.step()
        workload = self.workload_m.step()
        temp, norm_temp, wet_bulb, norm_wet_bulb = self.weather_m.step()
        ci_i, ci_i_future, ci_i_denorm = self.ci_m.step()
        
        self.current_hour = hour

        # Update environment states with new values from managers
        self._update_environments(workload, temp, wet_bulb, ci_i_denorm, ci_i_future, day, hour)

        # Create observations for the next step based on updated environment states
        next_workload = self.workload_m.get_next_workload()
        next_out_temperature = self.weather_m.get_next_temperature()

        queue_status = self.ls_info['ls_norm_tasks_in_queue']
        ci_i_past = self.ci_m.get_n_past_ci(n=16)
        oldest_task_age = self.ls_info['ls_oldest_task_age']
        average_task_age = self.ls_info['ls_average_task_age']
        ls_task_age_histogram = self.ls_info['ls_task_age_histogram']
        
        next_n_out_temperature = self.weather_m.get_n_next_temperature(n=32)

        
        self.ls_state = self._create_ls_state(t_i, workload, queue_status, ci_i, ci_i_future, ci_i_past, next_workload, norm_temp, next_out_temperature, next_n_out_temperature, oldest_task_age, average_task_age, ls_task_age_histogram, self.ls_info)
        self.dc_state = self._create_dc_state(workload, next_workload, norm_temp, next_out_temperature)
        self.bat_state = self._create_bat_state(workload, next_workload, ci_i, norm_temp)

        # Populate observation dictionary based on updated states
        obs = self._populate_observation_dict()

        # Calculate rewards for all agents based on the updated state
        location = self.location
        ci_i_12_hours = self.ci_m.get_n_future_ci(n=4*12)
        # if terminal:
            # print(f'Terminal state reached at day {day} and hour {hour}')
            
        reward_params = self._calculate_reward_params(workload, temp, ci_i, ci_i_future, day, hour, terminal, location, ci_i_12_hours)
        self.ls_reward, self.dc_reward, self.bat_reward = self.calculate_reward(reward_params)

        # Update rewards, terminations, and truncations for each agent
        self._update_reward_and_termination(rew, terminateds, truncateds)

        # Populate info dictionary with additional information
        info = self._populate_info_dict(reward_params)

        # Update the self.infos dictionary, similar to how it's done in the reset method
        self.infos = {
            'agent_ls': self.ls_info,
            'agent_dc': self.dc_info,
            'agent_bat': self.bat_info,
            '__common__': {
                'time': t_i,
                'workload': workload,
                'weather': temp,
                'ci': ci_i,
                'ci_future': ci_i_future,
                'states': {
                    'agent_ls': self.ls_state,
                    'agent_dc': self.dc_state,
                    'agent_bat': self.bat_state
                }
            }
        }


        # Handle termination if the episode ends
        if terminal:
            self._handle_terminal(terminateds, truncateds)

        # # print the max and the min value of the state
        # if self.max_value_state < np.max(obs['agent_ls']):
        #     self.max_value_state = np.max(obs['agent_ls'])
        #     print(f'Max value of the state: {self.max_value_state}')
        #     self.max_value_state = np.max(obs['agent_ls'])
        # if self.min_value_state > np.min(obs['agent_ls']):
        #     self.min_value_state = np.min(obs['agent_ls'])
        #     print(f'Min value of the state: {self.min_value_state}')
        #     self.min_value_state = np.min(obs['agent_ls'])

        return obs['agent_ls'], rew['agent_ls'], terminateds['agent_ls'], truncateds['agent_ls'], info


    def _perform_actions(self, action_dict):
        """Execute actions for each agent and update their respective environments."""
        # Load shifting agent
        if "agent_ls" in self.agents:
            action = action_dict
        else:
            action = self.base_agents["agent_ls"].do_nothing_action()

        # call step method of the load shifting environment with the action and the workload for the rest of the day
        # workload_rest_day = self.workload_m.get_n_next_workloads(n=int((24 - self.current_hour) / 0.25))
        self.ls_state, _, self.ls_terminated, self.ls_truncated, self.ls_info = self.ls_env.step(action)

        # Data center agent
        if "agent_dc" in self.agents:
            action = action_dict["agent_dc"]
        else:
            action = self.base_agents["agent_dc"].do_nothing_action()
        self.dc_env.set_shifted_wklds(self.ls_info['ls_shifted_workload'])
        self.dc_state, _, self.dc_terminated, self.dc_truncated, self.dc_info = self.dc_env.step(action)

        # Battery agent
        if "agent_bat" in self.agents:
            action = action_dict["agent_bat"]
        else:
            action = self.base_agents["agent_bat"].do_nothing_action()
        self.bat_env.set_dcload(self.dc_info['dc_total_power_kW'] / 1e3)
        self.bat_state, _, self.bat_terminated, self.bat_truncated, self.bat_info = self.bat_env.step(action)


    def _update_environments(self, workload, temp, wet_bulb, ci_i_denorm, ci_i_future, current_day, current_hour):
        """Update the environment states based on the manager's outputs."""
        self.ls_env.update_workload(workload)
        self.ls_env.update_current_date(current_day, current_hour)
        self.dc_env.set_ambient_temp(temp, wet_bulb)
        self.bat_env.update_ci(ci_i_denorm, ci_i_future[0])


    def _populate_observation_dict(self):
        """Generate the observation dictionary for all agents."""
        obs = {}
        if "agent_ls" in self.agents:
            obs['agent_ls'] = self.ls_state
        if "agent_dc" in self.agents:
            obs['agent_dc'] = self.dc_state
        if "agent_bat" in self.agents:
            obs['agent_bat'] = self.bat_state
        return obs


    def _calculate_reward_params(self, workload, temp, ci_i, ci_i_future, day, hour, terminal, location=None, ci_i_12_hours=None):
        """Create the parameters needed to calculate rewards."""
        return {
            **self.bat_info, **self.ls_info, **self.dc_info,
            "outside_temp": temp, "day": day, "hour": hour,
            "norm_CI": ci_i_future[0], "forecast_CI": ci_i_future,
            "isterminal": terminal, "location": location, 
            "ci_i_12_hours": ci_i_12_hours
        }


    def _update_reward_and_termination(self, rew, terminateds, truncateds):
        """Update the rewards, termination, and truncation flags for all agents."""
        if "agent_ls" in self.agents:
            rew["agent_ls"] = self.ls_reward
            terminateds["agent_ls"] = self.ls_terminated
            truncateds["agent_ls"] = self.ls_truncated
        if "agent_dc" in self.agents:
            rew["agent_dc"] = self.dc_reward
            terminateds["agent_dc"] = self.dc_terminated
            truncateds["agent_dc"] = self.dc_truncated
        if "agent_bat" in self.agents:
            rew["agent_bat"] = self.bat_reward
            terminateds["agent_bat"] = self.bat_terminated
            truncateds["agent_bat"] = self.bat_truncated


    def _populate_info_dict(self, reward_params):
        """Generate the info dictionary for all agents and common info."""
        info = {
            "agent_ls": {**self.dc_info, **self.ls_info, **self.bat_info, **reward_params},
            "agent_dc": {**self.dc_info, **self.ls_info, **self.bat_info, **reward_params},
            "agent_bat": {**self.dc_info, **self.ls_info, **self.bat_info, **reward_params},
            "__common__": reward_params
        }
        return info


    def _handle_terminal(self, terminateds, truncateds):
        """Handle the terminal state of the environment."""
        terminateds["__all__"] = False
        truncateds["__all__"] = True
        for agent in self.agents:
            truncateds[agent] = True
    
    
    def calculate_reward(self, params):
        """
        Calculate the individual reward for each agent.

        Args:
            params (dict): Dictionary of parameters to calculate the reward.

        Returns:
            ls_reward (float): Individual reward for the load shifting agent.
            dc_reward (float): Individual reward for the data center agent.
            bat_reward (float): Individual reward for the battery agent.
        """

        ls_reward = self.ls_reward_method(params)
        dc_reward = self.dc_reward_method(params)
        bat_reward = self.bat_reward_method(params)
        return ls_reward, dc_reward, bat_reward


    def render(self):
        """
        Render the environment.
        """
        pass


    def close(self):
        """
        Close the environments.
        """
        if hasattr(self, 'ls_env') and self.ls_env is not None:
            self.ls_env.close()
        if hasattr(self, 'dc_env') and self.dc_env is not None:
            self.dc_env.close()
        if hasattr(self, 'bat_env') and self.bat_env is not None:
            self.bat_env.close()
            
        
    def get_avail_actions(self):
        """
        Get the available actions for the agents.

        Returns:
            list: List of available actions for each agent.
        """
        if self.discrete:  # pylint: disable=no-member
            avail_actions = []
            for agent_id in range(self.n_agents):  # pylint: disable=no-member
                avail_agent = self.get_avail_agent_actions(agent_id)
                avail_actions.append(avail_agent)
            return avail_actions
        else:
            return None


    def get_avail_agent_actions(self, agent_id):
        """
        Get the available actions for a specific agent.

        Args:
            agent_id (int): Agent ID.

        Returns:
            list: List of available actions for the agent.
        """
        return [1] * self.action_space[agent_id].n
    
    
    def state(self):
        """
        Get the state of the environment.

        Returns:
            np.ndarray: State of the environment.
        """
        states = tuple(
            self.scenario.observation(  # pylint: disable=no-member
                self.world.agents[self._index_map[agent]], self.world  # pylint: disable=no-member
            ).astype(np.float32)
            for agent in self.possible_agents  # pylint: disable=no-member
        )
        return np.concatenate(states, axis=None)
    
    def get_hierarchical_variables(self):
        return self.datacenter_capacity, self.workload_m.get_current_workload(), self.weather_m.get_current_weather(), self.ci_m.get_current_ci()  # pylint: disable=no-member
        
    def set_hierarchical_workload(self, workload):
        self.workload_m.set_current_workload(workload)
        
    def get_available_capacity(self, time_steps: int) -> float:
        """
        Calculate the available capacity of the datacenter over the next time_steps.

        Args:
            time_steps (int): Number of 15-minute time steps to consider.

        Returns:
            float: The available capacity in MW.
        """
        # Initialize the available capacity
        available_capacity = 0

        # Retrieve the current workload and the datacenter's total capacity
        current_step = self.workload_m.time_step
        max_capacity = self.datacenter_capacity_mw

        # Calculate the remaining capacity over each time step
        for step in range(1, time_steps + 1):
            # Make sure we're not exceeding the total steps available in workload data
            if current_step + step < len(self.workload_m.cpu_smooth):
                current_workload = self.workload_m.cpu_smooth[current_step + step]
                remaining_capacity = (1 - current_workload)*max_capacity
                available_capacity += max(0, remaining_capacity)
                # print(f"Step: {step}, Current Workload: {current_workload}, Remaining Capacity: {remaining_capacity}, Available Capacity: {available_capacity}")

        # Return the average capacity over the given time steps
        return available_capacity