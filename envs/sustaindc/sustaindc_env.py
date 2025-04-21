import os
import sys
import random
import datetime
from collections import deque
from typing import Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces

from utils.make_envs import make_bat_fwd_env, make_dc_env, make_ls_env
from utils.managers import CI_Manager, ElectricityPrice_Manager, Time_Manager, Weather_Manager
from utils.utils_cf import get_energy_variables, get_init_day, obtain_paths

from ..env_config import EnvConfig

MAX_WAIT_TIMESTEPS = 4 * 8  # 8 hours, with 15-minute intervals = 32 timesteps

class SustainDC(gym.Env):
    def __init__(self, env_config):
        '''
        Initialize the SustainDC environment.

        Args:
            env_config (dict): Dictionary containing parameters as defined in 
                               EnvConfig above.
        '''
        super().__init__()

        # Initialize the environment config
        env_config = EnvConfig(env_config)
        self.env_config = env_config

        self.dc_id = env_config['dc_id']
        self.network_cost_per_gb = env_config.get('network_cost_per_gb', 0.0)  # Default to 0 if missing

        # Create environments and agents
        self.agents = env_config['agents']
        self.rbc_agents = env_config.get('rbc_agents', [])
        
        self.location = env_config['location']
                
        self.max_bat_cap_Mw = env_config['max_bat_cap_Mw']
        
        self.datacenter_capacity_mw = env_config['datacenter_capacity_mw']
        self.total_cores = env_config['total_cores']
        self.total_gpus = env_config['total_gpus']
        self.total_mem = env_config['total_mem']
        
        self.dc_config_file = env_config['dc_config_file']
        self.timezone_shift = env_config['timezone_shift']
        
        # Assign month according to worker index, if available
        if hasattr(env_config, 'worker_index'):
            self.month = int((env_config.worker_index - 1) % 12)
        else:
            self.month = env_config.get('month')

        self.evaluation_mode = env_config['evaluation']

        self._agent_ids = set(self.agents)
        
        n_vars_energy, n_vars_battery = 0, 0  # For partial observability (for p.o.)
        n_vars_ci = 8

        # **✅ Pick a random simulation year for variability**
        self.simulation_year = None 

        # Get resource capacities from config, provide default values if not specified
        self.total_cpus = env_config.get('total_cpus', 2000)
        self.total_gpus = env_config.get('total_gpus', 220)
        self.total_mem = env_config.get('total_mem', 2000)

        # Set available resources equal to total at initialization
        self.available_cpus = self.total_cpus
        self.available_gpus = self.total_gpus
        self.available_mem = self.total_mem

        # self.ls_env = make_ls_env(month=self.month, test_mode=self.evaluation_mode, n_vars_ci=n_vars_ci, 
        #                           n_vars_energy=n_vars_energy, n_vars_battery=n_vars_battery, queue_max_len=1000)
        self.dc_env, _ = make_dc_env(month=self.month, location=self.location, dc_memory=self.available_mem, max_bat_cap_Mw=self.max_bat_cap_Mw, use_ls_cpu_load=True, 
                                             datacenter_capacity_mw=self.datacenter_capacity_mw, dc_config_file=self.dc_config_file, add_cpu_usage=False)
        self.bat_env = make_bat_fwd_env(month=self.month, max_bat_cap_Mwh=self.dc_env.ranges['max_battery_energy_Mwh'], 
                                        max_dc_pw_MW=10, 
                                        dcload_max=10,
                                        dcload_min=1,
                                        n_fwd_steps=n_vars_ci, init_day=0)

        # self.bat_env.dcload_max = self.dc_env.power_ub_kW / 4  # Assuming 15 minutes timestep. Kwh
        # self.bat_env.dcload_min = self.dc_env.power_lb_kW / 4  # Assuming 15 minutes timestep. Kwh
        
        self._obs_space_in_preferred_format = True
        
        self.observation_space = []
        self.action_space = []

        # Get resource capacities from config, provide default values if not specified
        self.total_cores = env_config.get('total_cores', 2000)
        self.total_gpus = env_config.get('total_gpus', 220)
        self.total_mem = env_config.get('total_mem', 2000)

        # Set available resources equal to total at initialization
        self.available_cores = self.total_cores
        self.available_gpus = self.total_gpus
        self.available_mem = self.total_mem

        # Running & Pending Task Queues
        self.running_tasks = []
        self.pending_tasks = deque()
        self.current_time_task = 0

    def _create_dc_environment(self, env_config):
        """
        Creates the internal datacenter environment.
        """
        return SustainDC(env_config)
    
    def can_schedule(self, task):
        return (task.cores_req <= self.available_cores and
                task.gpu_req <= self.available_gpus and
                task.mem_req <= self.available_mem)
    
    def release_resources(self, current_time, logger):
        """
        Releases resources from completed tasks and logs their completion.
        """
        finished_tasks = [task for task in self.running_tasks if task.finish_time <= current_time]
        
        for task in finished_tasks:
            self.available_cores += task.cores_req
            self.available_gpus += task.gpu_req
            self.available_mem += task.mem_req
            task.sla_met = task.finish_time <= task.sla_deadline

            if logger:
                logger.info(f"[{current_time}] Task {task.job_name} finished, resources released. "
                            f"DC{self.dc_id} resources available: {self.available_cores:.3f} CPUs, "
                            f"{self.available_gpus:.3f} GPUs, {self.available_mem:.3f} MEM.")

        self.running_tasks = [task for task in self.running_tasks if task.finish_time > current_time]
        
        return finished_tasks


    def try_to_schedule_task(self, task, current_time, logger):
        """
        Attempts to schedule a task if sufficient resources are available.
        
        If the task is successfully scheduled:
            - Resources are allocated.
            - The task is moved to the running queue.
            - Logs the scheduling details.

        If resources are insufficient:
            - The task remains in the pending queue.
            - The system tracks how long the task has been waiting.
            - If the wait time exceeds MAX_WAIT_INTERVALS, the task is dropped.

        Returns:
            - `True` if the task was scheduled.
            - `False` if the task was re-added to the pending queue.
        """
        def log_info(msg):
            if logger:
                logger.info(msg)

        def log_warn(msg):
            if logger:
                logger.warning(msg)
            
        if self.can_schedule(task):
            # **Allocate resources**
            self.available_cores -= task.cores_req
            self.available_gpus -= task.gpu_req
            self.available_mem -= task.mem_req
            task.start_time = current_time
            task.finish_time = current_time + pd.Timedelta(minutes=task.duration)

            # **Compute network cost**
            # TODO: Implement network cost calculation based on task properties
            network_cost = -1

            # **Move task to running queue**
            self.running_tasks.append(task)

            log_info(f"[{current_time}] Task {task.job_name} scheduled successfully on DC{self.dc_id}. "
                    f"(CPU: {task.cores_req:.2f}, GPU: {task.gpu_req:.2f}, MEM: {task.mem_req:.2f}, "
                    f"Bandwidth: {task.bandwidth_gb:.4f}GB). "
                    f"Remaining: {self.available_cores:.2f} CPUs, {self.available_gpus:.2f} GPUs, "
                    f"{self.available_mem:.2f} GB MEM.")

            log_info(f"[{current_time}] Task {task.job_name} started and will finish at {task.finish_time}.")
            return True
        else:
            # **Task couldn't be scheduled - track wait time**
            task.increment_wait_intervals()
            if task.wait_intervals > MAX_WAIT_TIMESTEPS:
                log_warn(f"[{current_time}] Task {task.job_name} dropped after waiting too long. IGNORED")
                # Simulating the drop by not re-adding the task to the pending queue
            else:
                self.pending_tasks.append(task)  # Re-add task to queue for next cycle (The task is added to the end of the queue [right side])
                log_info(f"[{current_time}] Task {task.job_name} moved to pending queue of DC{self.dc_id} due to resource limits.")
            return False
            
    def set_seed(self, seed=None):
        """Set the random seed for the environment."""
        seed = seed or 1
        np.random.seed(seed)
        random.seed(seed)
        if torch is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self._seed_spaces()

    def _seed_spaces(self):
        """Seed the action and observation spaces."""
        if hasattr(self, 'action_space') and hasattr(self.action_space, 'seed'):
            self.action_space.seed(self.seed)
        if hasattr(self, 'observation_space') and hasattr(self.observation_space, 'seed'):
            self.observation_space.seed(self.seed)

    def reset(self, init_year=None, init_day=None, init_hour=None, seed=None, options=None):
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
        # Reset resources to initial values
        self.available_cores = self.total_cores
        self.available_gpus = self.total_gpus
        self.available_mem = self.total_mem

        self.running_tasks.clear()
        self.pending_tasks.clear()
        self.current_time_task = 0

        # **Reinitialize the managers with new paths**
        self.simulation_year = init_year
        self.t_m = Time_Manager(init_day, timezone_shift=self.timezone_shift)
        self.ci_manager = CI_Manager(location=self.location, simulation_year=self.simulation_year, timezone_shift=self.timezone_shift)
        self.weather_manager = Weather_Manager(location=self.location, simulation_year=self.simulation_year, timezone_shift=self.timezone_shift)
        self.price_manager = ElectricityPrice_Manager(location=self.location, simulation_year=self.simulation_year, timezone_shift=self.timezone_shift)

        # Reset termination and reward flags for all agents
        self.ls_terminated = self.dc_terminated = self.bat_terminated = False
        self.ls_truncated = self.dc_truncated = self.bat_truncated = False

        # Adjust based on local timezone
        local_init_day = (init_day + int((init_hour + self.timezone_shift) / 24)) % 365
        local_init_hour = (init_hour + self.timezone_shift) % 24
        self.current_hour = local_init_hour
        
        t_i = self.t_m.reset(init_day=local_init_day, init_hour=local_init_hour, seed=seed)
        temp, norm_temp, wet_bulb, norm_wet_bulb = self.weather_manager.reset(init_day=local_init_day, init_hour=local_init_hour, seed=seed)
        ci_i, ci_i_future, ci_i_denorm = self.ci_manager.reset(init_day=local_init_day, init_hour=local_init_hour, seed=seed)
        price_i = self.price_manager.reset(init_day=local_init_day, init_hour=local_init_hour, seed=seed)

        # Set the external ambient temperature to data center environment
        self.dc_env.set_ambient_temp(temp, wet_bulb)
        
        # Reset all the environments
        self.dc_state, self.dc_info = self.dc_env.reset()
        bat_s, self.bat_info = self.bat_env.reset()
                
        current_cpu_workload = 0.0  #self.workload_m.get_current_workload()
        current_gpu_workload = 0.0 
        self.dc_env.update_workload(current_cpu_workload)
        self.dc_env.update_gpu_workload(current_gpu_workload)

        # Update ci in the battery environment
        self.bat_env.update_ci(ci_i_denorm, ci_i_future[0])

        # States should be a dictionary with agent names as keys and their observations as values
        states = {}
        self.infos = {}
        # Update states and infos considering the agents defined in the environment config self.agents.
        if "agent_ls" in self.agents:
            states["agent_ls"] = self.ls_state
        if "agent_dc" in self.agents:
            states["agent_dc"] = self.dc_state
        if "agent_bat" in self.agents:
            states["agent_bat"] = self.bat_state

        # Prepare the infos dictionary with common and individual agent information
        self.infos = {
            'agent_dc': self.dc_info,
            'agent_bat': self.bat_info,
            '__common__': {
                'time': t_i,
                'cpu_workload': current_cpu_workload,
                'gpu_workload': current_gpu_workload,
                'weather': temp,
                'ci': ci_i,
                'ci_future': ci_i_future,
            }
        }
        
        
        # available_actions = None
        
        return states
    
    def step(self, action_dict, logger):
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
        day, hour, t_i = self.t_m.step()
        # workload = self.workload_m.step()
        temp, norm_temp, wet_bulb, norm_wet_bulb = self.weather_manager.step()
        ci_i, ci_i_future, ci_i_denorm = self.ci_manager.step()
        price_i = self.price_manager.step()
        
        # New logic for the task execution scheduler
        self.current_hour = hour

        # **1. Release resources from completed tasks**
        finished_tasks = self.release_resources(self.current_time_task, logger)

        num_tasks_assigned = 0
        routed_tasks_this_step = []
        # **2. Try scheduling pending tasks (FIFO order)**
        for _ in range(len(self.pending_tasks)):  # Process each task once
            task = self.pending_tasks.popleft()  # Take the first task from the queue
            scheduled = self.try_to_schedule_task(task, self.current_time_task, logger)  # Let schedule_task() handle scheduling or re-queuing
            if scheduled:
                num_tasks_assigned += 1
                routed_tasks_this_step.append(task)


        # **3. Log resource utilization**
        used_cores = round(self.total_cores, 6) - round(self.available_cores, 6)
        used_gpu = round(self.total_gpus, 6) - round(self.available_gpus, 6)
        used_mem = round(self.total_mem, 6) - round(self.available_mem, 6)

        # Convert used_cores => HPC environment usage
        # e.g. self.dc_env.set_shifted_wklds(used_cores / self.total_cores) or similar
        # HPC environment can produce the final usage metrics
        # I need a value between 0 and 1 for the workload.
        # At this time, we are only focused on the cpu usage.
        cpu_workload = used_cores / self.total_cores
        gpu_workload = used_gpu / self.total_gpus
        # print(f"[{self.current_time_task}] DC:{self.dc_id} Running: {len(self.running_tasks)}, Pending: {len(self.pending_tasks)}")
        if logger:
            logger.info(f"[{self.current_time_task}] DC:{self.dc_id} Running: {len(self.running_tasks)}, Pending: {len(self.pending_tasks)}")

        # Update environment states with new values from managers
        self._update_environments(cpu_workload, gpu_workload, temp, wet_bulb, ci_i_denorm, ci_i_future, day, hour)

        # Create observations for the next step based on updated environment states
        # Populate observation dictionary based on updated states
        obs = self._populate_observation_dict()

        # Update the self.infos dictionary, similar to how it's done in the reset method
        self.infos = {
            'agent_dc': self.dc_info,
            'agent_bat': self.bat_info,
            '__common__': {
                'time': t_i,
                # 'workload': workload,
                'weather': temp,
                'ci': ci_i_denorm,
                'price_USD_kwh': price_i,
                'routed_tasks_this_step': routed_tasks_this_step,
            }
        }
        
        sla_stats = {
                    "met": 0,
                    "violated": 0,
                    }

        for task in finished_tasks:
            if task.sla_met:
                sla_stats["met"] += 1
            else:
                sla_stats["violated"] += 1
        
        # Calculate percentages
        cpu_util = used_cores / self.total_cores
        gpu_util = used_gpu / self.total_gpus
        mem_util = used_mem / self.total_mem

        # Compute number of tasks
        running_count = len(self.running_tasks)
        pending_count = len(self.pending_tasks)

        # Estimate energy cost
        energy_kwh = self.bat_info.get("bat_total_energy_with_battery_KWh", 0)
        energy_cost_USD = energy_kwh * price_i/1000  # Assuming price_i is in $ per kWh
        energy_ci = self.bat_info.get("bat_avg_CI", 0)  # Average carbon intensity in gCO₂eq/kWh
        # Estimate CO₂ emissions, ci_i is in gCO₂eq/kWh
        carbon_emissions_kg = energy_kwh * energy_ci / 1000  # Convert gCO₂eq to kgCO₂eq

        # Add to common info
        self.infos["__common__"].update({
            "cpu_util_percent": cpu_util * 100,
            "gpu_util_percent": gpu_util * 100,
            "mem_util_percent": mem_util * 100,

            "running_tasks": running_count,
            "pending_tasks": pending_count,
            "tasks_assigned": num_tasks_assigned,

            # "dc_total_power_kW": self.dc_info.get("dc_total_power_kW", None),
            "energy_consumption_kwh": energy_kwh,

            "energy_cost_USD": energy_cost_USD,
            "carbon_emissions_kg": carbon_emissions_kg,
            "__sla__": sla_stats,
        })

        return obs, rew, terminateds, truncateds, self.infos


    def _perform_actions(self, action_dict):
        # Use fixed "do nothing" actions:
        # DO_NOTHING_LOAD_SHIFTING = 1
        DO_NOTHING_HVAC = 1
        DO_NOTHING_BATTERY = 2

        # For load shifting environment:
        # self.ls_state, _, self.ls_terminated, self.ls_truncated, self.ls_info = self.ls_env.step(DO_NOTHING_LOAD_SHIFTING)

        # For HVAC / data center environment:
        self.dc_state, _, self.dc_terminated, self.dc_truncated, self.dc_info = self.dc_env.step(DO_NOTHING_HVAC)

        # For battery environment:
        self.bat_env.set_dcload(self.dc_info['dc_total_power_kW'] / 1e3)
        self.bat_state, _, self.bat_terminated, self.bat_truncated, self.bat_info = self.bat_env.step(DO_NOTHING_BATTERY)



    def _update_environments(self, workload, gpu_workload, temp, wet_bulb, ci_i_denorm, ci_i_future, current_day, current_hour):
        """Update the environment states based on the manager's outputs."""
        # self.ls_env.update_workload(workload)
        # self.ls_env.update_current_date(current_day, current_hour)
        self.dc_env.set_ambient_temp(temp, wet_bulb)
        self.dc_env.update_workload(workload)
        self.dc_env.update_gpu_workload(gpu_workload)
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

    
    def state(self):
        """
        Get the state of the environment.

        Returns:
            np.ndarray: State of the environment.
        """
        print('Calling the method state() of SustainDC')
        states = tuple(
            self.scenario.observation(  # pylint: disable=no-member
                self.world.agents[self._index_map[agent]], self.world  # pylint: disable=no-member
            ).astype(np.float32)
            for agent in self.possible_agents  # pylint: disable=no-member
        )
        return np.concatenate(states, axis=None)

    def _update_current_time_task(self, current_time):
        """
        Update the current time of the task.
        """
        self.current_time_task = current_time

