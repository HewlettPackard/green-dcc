import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

class CarbonLoadEnv(gym.Env):
    def __init__(
        self,
        env_config = {},
        future=True,
        n_vars_ci=4,
        flexible_workload_ratio=0.2,
        n_vars_energy=0,
        n_vars_battery=1,
        test_mode=False,
        queue_max_len=500,
        ):
        """Creates load shifting envrionemnt

        Args:
            env_config (dict, optional): Customizable environment confing. Defaults to {}.
            future (bool, optional): To include CI forecast to the observation. Defaults to True.
            future_steps (int, optional): Number of time steps in the future in the forecast. Defaults to 4.
            flexible_workload_ratio (float, optional): Percentage of flexible workload. Defaults to 0.1.
            n_vars_energy (int, optional): Additional number of energy variables. Defaults to 0.
            n_vars_battery (int, optional): Additional number of variables from the battery. Defaults to 1.
            test_mode (bool, optional): Used for evaluation of the model. Defaults to False.
        """
        assert flexible_workload_ratio < 0.9, "flexible_workload_ratio should be lower than 0.9"
        self.flexible_workload_ratio = flexible_workload_ratio
        
        # Define a single continuous action space: [-1, 1]
        # -1: Defer all shiftable tasks, 0: Do nothing, 1: Process all DTQ tasks
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        
        # State: [Sin(h), Cos(h), Sin(day_of_year), Cos(day_of_year), self.ls_state, ci_i_future (n_vars_ci), var_to_LS_energy (n_vars_energy), batSoC (n_vars_battery)], 
        # self.ls_state = [current_workload, queue status]
        if future:
            self.observation_space = spaces.Box(
                low=0,
                high=1,
                shape=(6 + n_vars_ci + n_vars_energy + n_vars_battery,),
                dtype=np.float32,
            )
        else:
            self.observation_space = spaces.Box(
                low=0,
                high=1,
                shape=(6 + n_vars_energy + n_vars_battery,),
                dtype=np.float32,
            )


        self.global_total_steps = 0
        self.test_mode = test_mode
        self.time_steps_day = 96
        # self.load_to_assign = 3 * flexible_workload_ratio
        # self.day_workload = 0
        self.workload = 0
        
        # Initialize the queue to manage individual delayed tasks
        self.tasks_queue = []  # A list to hold individual tasks
        self.queue_max_len = queue_max_len

    def reset(self, *, seed=None, options=None):
        """
        Reset `CarbonLoadEnv` to initial state.

        Returns:
            observations (List[float]): Current state of the environmment
            info (dict): A dictionary that containing additional information about the environment state
        """
        self.global_total_steps = 0
        self.tasks_queue = []  # Clear the task queue
        self.current_hour = 0.0
        self.day = 0
        
        # Queue status - length of the task queue
        current_workload = self.workload
        queue_length = 0
        
        state = np.asarray(np.hstack(([current_workload, queue_length/self.queue_max_len])), dtype=np.float32)
        
        info = {"load": self.workload,
                "action": -1,
                "info_load_left": 0,
                "ls_tasks_dropped": 0,
                "ls_tasks_in_queue": 0}
        
        return state, info

    def step(self, action):
        """
        Makes an environment step in `CarbonLoadEnv`.

        Args:
            action (float): Continuous action between -1 and 1.
                            -1: Defer all shiftable tasks.
                             1: Process tasks from the DTQ to maximize utilization.
                             Values between -1 and 0 defer a fraction of tasks, and values between 0 and 1 process a fraction of tasks in the DTQ.

        Returns:
            state (List[float]): Current state of the environment.
            reward (float): Reward value.
            done (bool): A boolean signaling if the episode has ended.
            info (dict): A dictionary containing additional information about the environment state.
        """
        self.current_hour += 0.25

        shiftable_tasks_percentage = self.flexible_workload_ratio
        non_shiftable_tasks_percentage = 1 - shiftable_tasks_percentage

        non_shiftable_tasks = int(math.ceil(self.workload * non_shiftable_tasks_percentage * 100))
        shiftable_tasks     = int(math.floor(self.workload * shiftable_tasks_percentage * 100))
        tasks_dropped = 0  # Track the number of dropped tasks
        
        action_value = action[0]  # Single continuous action

        if action_value < 0:  # Defer tasks
            defer_ratio = abs(action_value)  # Convert the negative action to a positive defer ratio (0 to 1)
            tasks_to_defer = int(shiftable_tasks * defer_ratio)

            # Attempt to queue deferred tasks, tracking any that are dropped
            timestamp = self.current_hour  # Current timestamp
            for _ in range(tasks_to_defer):
                if len(self.tasks_queue) < self.queue_max_len:
                    self.tasks_queue.append({'timestamp': timestamp, 'utilization': 1})
                else:
                    tasks_dropped += 1

            # Update current utilization with only non-shiftable tasks plus remaining non-deferred shiftable tasks
            self.current_utilization = (non_shiftable_tasks + (shiftable_tasks - tasks_to_defer)) / 100

        elif action_value > 0:  # Process tasks from the DTQ
            available_capacity = 100 - (non_shiftable_tasks + shiftable_tasks)  # Remaining capacity
            if available_capacity >= 1: # At least 1 task can be processed
                process_ratio = action_value  # Positive action is the process ratio (0 to 1)
                tasks_to_process = int(available_capacity * process_ratio)  # Fill available capacity
                actual_tasks_to_process = min(len(self.tasks_queue), tasks_to_process)  # Don't process more than available

                # Process tasks from the DTQ
                for _ in range(actual_tasks_to_process):
                    if self.tasks_queue:
                        self.tasks_queue.pop(0)  # Remove a task from the queue for processing

                # Update current utilization to include processed DTQ tasks
                self.current_utilization = (non_shiftable_tasks + shiftable_tasks + actual_tasks_to_process) / 100
            else:
                self.current_utilization = (non_shiftable_tasks + shiftable_tasks) / 100

        else:  # action_value == 0, Do nothing
            self.current_utilization = (non_shiftable_tasks + shiftable_tasks) / 100
            
        self.global_total_steps += 1
        
        original_workload = self.workload

        if self.current_hour % (24*4) == 0:   # Penalty for queued tasks at the end of the day
            self.tasks_queue = []
            print(f'Checked that the tasks_queue is cleaned every 24 hours at {self.current_hour}')
        
        
        if self.current_hour >= 24:
            self.current_hour = 0
            
        reward = 0 
        
        tasks_in_queue = len(self.tasks_queue)
                
        info = {"ls_original_workload": original_workload,
                "ls_shifted_workload": self.current_utilization, 
                "ls_action": action, 
                "ls_norm_load_left": 0,
                "ls_unasigned_day_load_left": 0,
                "ls_penalty_flag": 0,
                'ls_tasks_in_queue': tasks_in_queue, 
                'ls_tasks_dropped': tasks_dropped,
                'ls_current_hour': self.current_hour}


        #Done and truncated are managed by the main class, implement individual function if needed
        truncated = False
        done = False
        
        if self.current_utilization > 1 or self.current_utilization < 0:
            print('WARNING, the utilization is out of bounds')
        state = np.asarray(np.hstack(([self.current_utilization, tasks_in_queue/self.queue_max_len])), dtype=np.float32)
        
        return state, reward, done, truncated, info 
        
    def update_workload(self, workload):
        """
        Makes an environment step in`BatteryEnvFwd.

        Args:
            workload (float): Workload assigned at the current time step
        """
        if workload < 0 or workload > 1:
            print('WARNING, the workload is out of bounds')
            # Raise an error if the workload is out of bounds
            raise ValueError("The workload should be between 0 and 1")
        self.workload = workload