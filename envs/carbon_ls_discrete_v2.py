import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from collections import deque


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
        initialize_queue_at_reset=False,
        num_action_levels=5  # Number of discrete action levels
        ):
        """Creates a discrete load shifting environment.

        Args:
            env_config (dict, optional): Customizable environment config. Defaults to {}.
            future (bool, optional): To include CI forecast to the observation. Defaults to True.
            flexible_workload_ratio (float, optional): Percentage of flexible workload. Defaults to 0.2.
            num_action_levels (int, optional): Number of discrete action levels. Defaults to 5.
        """
        assert flexible_workload_ratio < 1, "flexible_workload_ratio should be lower than 1.0 (100%)"
        self.flexible_workload_ratio = flexible_workload_ratio
        
        self.shiftable_tasks_percentage = self.flexible_workload_ratio
        self.non_shiftable_tasks_percentage = 1 - self.flexible_workload_ratio
        
        # Define a discrete action space: [0, num_action_levels - 1]
        self.num_action_levels = num_action_levels
        self.action_space = spaces.Discrete(self.num_action_levels)
        
        # State: same as before
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(17,), dtype=np.float32)

        self.global_total_steps = 0
        self.test_mode = test_mode
        self.time_steps_day = 96
        self.workload = 0

        self.queue_max_len = queue_max_len
        self.tasks_queue = deque(maxlen=self.queue_max_len)
        self.initialize_queue_at_reset = initialize_queue_at_reset

        # Initialize previous computed workload variable
        self.previous_computed_workload = 0.0

    # Calculate task age histogram
    def get_task_age_histogram(self, tasks_queue, current_day, current_hour):
        age_bins = [0, 6, 12, 18, 24, np.inf]  # Age bins in hours
        task_ages = [
            (current_day - task['day']) * 24 + (current_hour - task['hour'])
            for task in self.tasks_queue
        ]
        histogram, _ = np.histogram(task_ages, bins=age_bins)
        normalized_histogram = histogram / max(len(self.tasks_queue), 1)  # Avoid division by zero
        
        # Replace with a binary value for each bin if there are any tasks in each bin
        normalized_histogram = np.where(normalized_histogram > 0, 1, 0)
        
        return normalized_histogram  # Returns an array of proportions
    
    def reset(self, *, seed=None, options=None):
        """
        Reset `CarbonLoadEnv` to initial state.

        Returns:
            observations (List[float]): Current state of the environmment
            info (dict): A dictionary that containing additional information about the environment state
        """
        self.global_total_steps = 0
        self.tasks_queue.clear()
        
        if self.initialize_queue_at_reset:
            # Initialize the task queue with tasks of varying ages
            initial_queue_length = np.random.randint(1, self.queue_max_len // 4)

            # Generate random task ages between 0 and 24 hours
            # max_task_age = 24  # Maximum age in hours
            # task_ages = np.random.random_integers(0, max_task_age*4, initial_queue_length)/4
            
            # Generate task ages using an exponential distribution
            max_task_age = 24  # Maximum age in hours
            # Set the rate parameter (lambda) for the exponential distribution
            lambda_param = 1.0 / 4.0  # Mean age of 6 hours (adjust as needed)
            task_ages = np.round(np.random.exponential(scale=1.0 / lambda_param, size=initial_queue_length) * 4)/4

            # Cap the task ages at max_task_age
            task_ages = np.clip(task_ages, 0, max_task_age)

            # Sort the task ages in descending order
            task_ages = np.sort(task_ages)[::-1]

            for age in task_ages:
                # Compute the day and hour when the task was added
                task_day = self.current_day
                task_hour = self.current_hour - age

                # Adjust day and hour if task_hour is negative
                while task_hour < 0:
                    task_hour += 24
                    task_day -= 1  # Task was added on a previous day

                # Ensure task_day is non-negative
                if task_day < 0:
                    task_day = 0
                    task_hour = 0  # Reset to the earliest possible time

                # Create the task with its timestamp
                task = {'day': task_day, 'hour': task_hour, 'utilization': 1}
                self.tasks_queue.append(task)
        else:
            # Start with an empty task queue
            pass

        # Calculate queue_length, oldest_task_age, average_task_age
        tasks_in_queue = len(self.tasks_queue)
        if tasks_in_queue > 0:
            task_ages = [
                (self.current_day - task['day']) * 24 + (self.current_hour - task['hour'])
                for task in self.tasks_queue
            ]
            oldest_task_age = max(task_ages)
            average_task_age = sum(task_ages) / len(task_ages)
        else:
            oldest_task_age = 0.0
            average_task_age = 0.0

        task_age_histogram = self.get_task_age_histogram(self.tasks_queue, self.current_day, self.current_hour)
    
        # Compute state
        current_workload = self.workload  # Ensure self.workload is set appropriately
        state = np.asarray(np.hstack(([current_workload,
                                tasks_in_queue/self.queue_max_len,
                                oldest_task_age/24,
                                average_task_age/24,
                                task_age_histogram])), dtype=np.float32)
        

        # Initialize previous computed workload to 0 on reset
        self.previous_computed_workload = 0.0
    
        info = {"ls_original_workload": self.workload,
                "ls_shifted_workload": self.workload, 
                "ls_previous_computed_workload": self.previous_computed_workload,
                "ls_action": 0, 
                "info_load_left": 0,
                'ls_queue_max_len': self.queue_max_len,
                "ls_tasks_dropped": 0,
                "ls_tasks_in_queue": 0, 
                "ls_norm_tasks_in_queue": 0,
                'ls_tasks_processed': 0,
                'ls_enforced': False,
                'ls_oldest_task_age': oldest_task_age,
                'ls_average_task_age': average_task_age,
                'ls_overdue_penalty': 0,
                'ls_task_age_histogram': task_age_histogram,}
        
        
        return state, info


    # def step(self, action, workload_rest_day=0):
    #     """
    #     Makes an environment step in `CarbonLoadEnv`.

    #     Args:
    #         action (float): Continuous action between 0 and 1.
    #                         The desired computed workload as a fraction of capacity.
    #     """
    #     assert self.action_space.contains(action), f"Action {action} is not in the action space {self.action_space}"
        
    #     # Map the discrete action to a workload level between 0 and 1
    #     action_value = action / (self.num_action_levels - 1)
        
    #     capacity = 100  # Total capacity in tasks per time step
    #     desired_tasks_to_process = int(action_value * capacity) #- non_shiftable_tasks
        
    #     # Calculate mandatory tasks
    #     non_shiftable_tasks = int(round(self.workload * self.non_shiftable_tasks_percentage * capacity))

    #     # Handle overdue tasks
    #     overdue_tasks = [task for task in self.tasks_queue if (self.current_day - task['day']) * 24 + (self.current_hour - task['hour']) > 24]
    #     overdue_penalty = len(overdue_tasks)

    #     # Calculate initial available capacity
    #     available_capacity = capacity - non_shiftable_tasks  # Limit to remaining capacity after non-shiftable tasks

    #     # Process overdue tasks if there's capacity
    #     overdue_tasks_to_process = 0
    #     if available_capacity > 0 and len(overdue_tasks) > 0:
    #         overdue_task_count = len(overdue_tasks)
    #         tasks_that_can_be_processed = min(overdue_task_count, available_capacity)
    #         overdue_tasks_to_process = tasks_that_can_be_processed

    #         for task in overdue_tasks[:tasks_that_can_be_processed]:
    #             self.tasks_queue.remove(task)        
        
    #     # Total mandatory tasks
    #     mandatory_tasks = non_shiftable_tasks + overdue_tasks_to_process

    #     # Ensure that desired_tasks_to_process is at least the mandatory tasks
    #     under_desired = 0
    #     if desired_tasks_to_process < mandatory_tasks:
    #         # The agent must process at least the mandatory tasks
    #         under_desired = mandatory_tasks - desired_tasks_to_process
    #         desired_tasks_to_process = mandatory_tasks  # Adjust desired tasks to mandatory

    #     # Ensure we do not exceed capacity
    #     if desired_tasks_to_process > capacity:
    #         desired_tasks_to_process = capacity

    #     # After processing mandatory tasks, capacity remaining
    #     capacity_remaining = capacity - mandatory_tasks

    #     # Remaining desired tasks after mandatory tasks
    #     remaining_desired_tasks = desired_tasks_to_process - mandatory_tasks

    #     tasks_processed = mandatory_tasks

    #     # Process shiftable tasks from current workload
    #     shiftable_tasks = int(round(self.workload * self.shiftable_tasks_percentage * capacity))

    #     tasks_to_process_from_shiftable = 0
    #     if remaining_desired_tasks > 0 and capacity_remaining > 0:
    #         tasks_to_process_from_shiftable = min(shiftable_tasks, remaining_desired_tasks, capacity_remaining)
    #         tasks_processed += tasks_to_process_from_shiftable
    #         capacity_remaining -= tasks_to_process_from_shiftable
    #         remaining_desired_tasks -= tasks_to_process_from_shiftable

    #     # Process tasks from the queue
    #     tasks_to_process_from_queue = 0
    #     if remaining_desired_tasks > 0 and capacity_remaining > 0:
    #         tasks_to_process_from_queue = min(len(self.tasks_queue), remaining_desired_tasks, capacity_remaining)
    #         for _ in range(int(tasks_to_process_from_queue)):
    #             self.tasks_queue.popleft()
    #         tasks_processed += tasks_to_process_from_queue
    #         capacity_remaining -= tasks_to_process_from_queue
    #         remaining_desired_tasks -= tasks_to_process_from_queue

    #     # Defer remaining shiftable tasks
    #     tasks_to_defer = shiftable_tasks - tasks_to_process_from_shiftable
    #     available_queue_space = self.queue_max_len - len(self.tasks_queue)
    #     tasks_to_add_to_queue = min(tasks_to_defer, available_queue_space)
    #     self.tasks_queue.extend(
    #         [{'day': self.current_day, 'hour': self.current_hour, 'utilization': 1}] * int(tasks_to_add_to_queue)
    #     )
    #     tasks_dropped = tasks_to_defer - tasks_to_add_to_queue  # Tasks that couldn't be deferred due to full queue

    #     # Update current utilization
    #     self.current_utilization = tasks_processed / capacity
        
    #     # if not self.initialize_queue_at_reset: # That means that we are on eval mode
    #         # self.current_utilization += non_shiftable_tasks / capacity
                    
    #     original_workload = self.workload
    #     tasks_in_queue = len(self.tasks_queue)
            
    #     reward = 0 
        
    #     # Calculate queue statistics
    #     tasks_in_queue = len(self.tasks_queue)
    #     if tasks_in_queue > 0:
    #         task_ages = [
    #             (self.current_day - task['day']) * 24 + (self.current_hour - task['hour'])
    #             for task in self.tasks_queue
    #         ]
    #         oldest_task_age = max(task_ages)
    #         average_task_age = sum(task_ages) / len(task_ages)
    #     else:
    #         oldest_task_age = 0.0
    #         average_task_age = 0.0

    #     task_age_histogram = self.get_task_age_histogram(self.tasks_queue, self.current_day, self.current_hour)



    #     info = {"ls_original_workload": original_workload,
    #             "ls_shifted_workload": self.current_utilization, 
    #             "ls_previous_computed_workload": self.previous_computed_workload,
    #             "ls_action": action_value, 
    #             "ls_norm_load_left": 0,
    #             "ls_unasigned_day_load_left": 0,
    #             "ls_penalty_flag": 0,
    #             'ls_queue_max_len': self.queue_max_len,
    #             'ls_tasks_in_queue': tasks_in_queue, 
    #             'ls_norm_tasks_in_queue': tasks_in_queue/self.queue_max_len,
    #             'ls_tasks_dropped': tasks_dropped,
    #             'ls_current_hour': self.current_hour,
    #             'ls_tasks_processed': tasks_processed,
    #             'ls_enforced': False,
    #             'ls_oldest_task_age': oldest_task_age/24,
    #             'ls_average_task_age': average_task_age/24,
    #             'ls_overdue_penalty': overdue_penalty,
    #             'ls_computed_tasks': int(self.current_utilization*100),
    #             'ls_task_age_histogram': task_age_histogram,}

    #     # Update the environment state
    #     self.global_total_steps += 1
    
    #     if info['ls_shifted_workload'] > 1 or info['ls_shifted_workload'] < 0:
    #         # Launch an exception
    #         raise ValueError("The utilization should be between 0 and 1")

    #     # if overdue_penalty > 0:
    #         # print(f'Overdue penalty: {overdue_penalty}; Tasks processed: {actual_tasks_processed}; Task in queue: {tasks_in_queue}; Oldest task age: {oldest_task_age}; Average task age: {average_task_age:.2f}')
            
    #     # Update the previous computed workload
    #     self.previous_computed_workload = self.current_utilization  # This stores the previous timestep's workload
        
    #     #Done and truncated are managed by the main class, implement individual function if needed
    #     truncated = False
    #     done = False
        
    #     if self.current_utilization > 1 or self.current_utilization < 0:
    #         print('WARNING, the utilization is out of bounds')
    #     state = np.asarray(np.hstack(([self.current_utilization,
    #                                    tasks_in_queue/self.queue_max_len,
    #                                    oldest_task_age/24,
    #                                    average_task_age/24,
    #                                    task_age_histogram])), dtype=np.float32)
        
    #     return state, reward, done, truncated, info 
    def step(self, action, workload_rest_day=0):
        """
        Performs an environment step in `CarbonLoadEnv`.

        Args:
            action (int): Discrete action representing the desired workload level.

        Returns:
            state (np.array): The next state.
            reward (float): The reward obtained.
            done (bool): Whether the episode is done.
            truncated (bool): Whether the episode was truncated.
            info (dict): Additional information.
        """
        # Ensure the action is valid
        assert self.action_space.contains(action), f"Action {action} is not in the action space {self.action_space}"

        # Map the discrete action to a workload level between 0 and 1
        action_value = action / (self.num_action_levels - 1)  # Assuming num_action_levels >= 2

        capacity = 100  # Total capacity in tasks per time step

        # Desired tasks to process, as per agent's action
        desired_tasks_to_process = int(action_value * capacity)

        # Calculate non-shiftable tasks from current workload
        non_shiftable_tasks = int(round(self.workload * self.non_shiftable_tasks_percentage * capacity))

        # Enqueue all shiftable tasks from current workload
        shiftable_tasks = int(round(self.workload * self.shiftable_tasks_percentage * capacity))

        # Add shiftable tasks to the queue
        available_queue_space = self.queue_max_len - len(self.tasks_queue)
        tasks_to_add_to_queue = min(shiftable_tasks, available_queue_space)
        self.tasks_queue.extend(
            [{'day': self.current_day, 'hour': self.current_hour, 'utilization': 1}] * tasks_to_add_to_queue
        )
        tasks_dropped = shiftable_tasks - tasks_to_add_to_queue  # Tasks that couldn't be deferred due to full queue

        # Identify overdue tasks in the queue (tasks older than 24 hours)
        overdue_tasks = [task for task in self.tasks_queue
                        if (self.current_day - task['day']) * 24 + (self.current_hour - task['hour']) > 24]
        overdue_penalty = len(overdue_tasks)

        # Initialize counters
        tasks_processed = 0
        capacity_remaining = capacity

        # Process overdue tasks first
        overdue_tasks_to_process = min(len(overdue_tasks), capacity_remaining)
        for _ in range(overdue_tasks_to_process):
            task = overdue_tasks.pop(0)
            self.tasks_queue.remove(task)
        tasks_processed += overdue_tasks_to_process
        capacity_remaining -= overdue_tasks_to_process

        # Process non-shiftable tasks
        non_shiftable_tasks_to_process = min(non_shiftable_tasks, capacity_remaining)
        tasks_processed += non_shiftable_tasks_to_process
        capacity_remaining -= non_shiftable_tasks_to_process

        # Total mandatory tasks processed
        mandatory_tasks_processed = overdue_tasks_to_process + non_shiftable_tasks_to_process

        # Adjust desired tasks to process
        desired_tasks_to_process = max(desired_tasks_to_process, mandatory_tasks_processed)
        desired_tasks_to_process = min(desired_tasks_to_process, capacity)

        # Remaining desired tasks after mandatory tasks
        remaining_desired_tasks = desired_tasks_to_process - mandatory_tasks_processed

        # Process tasks from the queue
        tasks_to_process_from_queue = min(len(self.tasks_queue), remaining_desired_tasks, capacity_remaining)
        for _ in range(tasks_to_process_from_queue):
            self.tasks_queue.popleft()
        tasks_processed += tasks_to_process_from_queue
        capacity_remaining -= tasks_to_process_from_queue

        # Update current utilization
        self.current_utilization = tasks_processed / capacity

        # Update tasks_in_queue
        tasks_in_queue = len(self.tasks_queue)

        # Calculate queue statistics
        if tasks_in_queue > 0:
            task_ages = [
                (self.current_day - task['day']) * 24 + (self.current_hour - task['hour'])
                for task in self.tasks_queue
            ]
            oldest_task_age = max(task_ages)
            average_task_age = sum(task_ages) / len(task_ages)
        else:
            oldest_task_age = 0.0
            average_task_age = 0.0

        # Get task age histogram
        task_age_histogram = self.get_task_age_histogram(self.tasks_queue, self.current_day, self.current_hour)

        # Prepare info dictionary
        info = {
            "ls_original_workload": self.workload,
            "ls_shifted_workload": self.current_utilization,
            "ls_previous_computed_workload": self.previous_computed_workload,
            "ls_action": action_value,
            "ls_norm_load_left": 0,
            "ls_unassigned_day_load_left": 0,
            "ls_penalty_flag": 0,
            'ls_queue_max_len': self.queue_max_len,
            'ls_tasks_in_queue': tasks_in_queue,
            'ls_norm_tasks_in_queue': tasks_in_queue / self.queue_max_len,
            'ls_tasks_dropped': tasks_dropped,
            'ls_current_hour': self.current_hour,
            'ls_tasks_processed': tasks_processed,
            'ls_enforced': False,
            'ls_oldest_task_age': oldest_task_age / 24,
            'ls_average_task_age': average_task_age / 24,
            'ls_overdue_penalty': overdue_penalty,
            'ls_computed_tasks': tasks_processed,
            'ls_task_age_histogram': task_age_histogram,
        }

        # Update the environment state
        self.global_total_steps += 1

        # Update the previous computed workload
        self.previous_computed_workload = self.current_utilization

        # Assume done and truncated are False for now
        truncated = False
        done = False

        # Update state
        state = np.asarray(np.hstack((
            [
                self.current_utilization,
                tasks_in_queue / self.queue_max_len,
                oldest_task_age / 24,
                average_task_age / 24,
                task_age_histogram
            ]
        )), dtype=np.float32)

        # Compute reward (to be defined)
        reward = 0  # Define your reward function here

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
    
    def update_current_date(self, current_day, current_hour):
        """
        Update the current hour in the environment.

        Args:
            current_hour (float): Current hour in the environment.
        """
        self.current_day = current_day
        self.current_hour = current_hour
