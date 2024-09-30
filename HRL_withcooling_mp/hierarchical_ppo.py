import os
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor

file_dir = os.path.dirname(__file__)
# append one level upper directory to python path
sys.path.append(file_dir + '/..')
# pylint: disable=C0301,C0303,C0103,C0209,C0116,C0413
import HRL_withcooling_mp.ppo as ppo_class
from utils.utils_cf import generate_node_connections

class HierarchicalPPO:
    
    def __init__(self, num_ll_policies, num_dc_policies,
                 obs_dim_hl, obs_dim_ll, obs_dim_dc,
                 action_dim_hl, action_dim_ll, action_dim_dc,
                 goal_dim_ll, goal_dim_dc,
                 hl_lr_actor, hl_lr_critic, ll_lr_actor, ll_lr_critic, dc_lr_actor, dc_lr_critic,
                 hl_gamma, ll_gamma, dc_gamma, hl_K_epochs, ll_K_epochs, dc_K_epochs,
                 eps_clip,
                 hl_has_continuous_action_space=True, ll_has_continuous_action_space=True, dc_has_continuous_action_space=False,
                 action_std_init=0.6,
                 high_policy_action_freq=4,
                 ll_policy_ids=None):

        # High-level policy
        self.high_policy = ppo_class.PPO(
            obs_dim_hl, action_dim_hl,
            hl_lr_actor, hl_lr_critic, hl_gamma, hl_K_epochs,
            eps_clip, hl_has_continuous_action_space, action_std_init
        )

        # Low-level policies (LS agents)
        self.low_policies = [
            ppo_class.PPO(
                state_dim + goal_dim, action_dim,
                ll_lr_actor, ll_lr_critic, ll_gamma, ll_K_epochs,
                eps_clip, ll_has_continuous_action_space, action_std_init
            )
            for state_dim, goal_dim, action_dim in zip(obs_dim_ll, goal_dim_ll, action_dim_ll)
        ]

        # DC agent policies
        self.dc_policies = [
            ppo_class.PPO(
                state_dim + goal_dim, action_dim,
                dc_lr_actor, dc_lr_critic, dc_gamma, dc_K_epochs,
                eps_clip, dc_has_continuous_action_space, action_std_init
            )
            for state_dim, goal_dim, action_dim in zip(obs_dim_dc, goal_dim_dc, action_dim_dc)
        ]

        self.high_policy_action_freq = high_policy_action_freq
        self.action_counter = 0
        self.goal = np.random.uniform(-1, 1, action_dim_hl)
        self.ll_policy_ids = ll_policy_ids

        
    def select_action(self, state):
        actions = {}
        
        # High-level actions
        if self.action_counter % self.high_policy_action_freq == 0:
            self.goal = self.high_policy.select_action(state['high_level_obs'])
        self.action_counter += 1
        
        actions['high_level_action'] = np.clip(self.goal, -1.0, 1.0)
        
        # Mapping high-level actions to low-level goals
        goal_list_ll = []  # Goals for LS agents
        goal_list_dc = []  # Goals for DC agents
        
        # Assuming generate_node_connections generates connections for both LS and DC agents
        node_connections = generate_node_connections(N=[i for i in range(len(self.low_policies))], E=self.goal)
        for _, edges in node_connections.items():
            # Extract goals for LS and DC agents
            ll_goal = [e[1] for e in edges]
            dc_goal = [e[1] for e in edges]  # Modify as per your goal structure
            goal_list_ll.append(ll_goal)
            goal_list_dc.append(dc_goal)
        
        # Low-level LS agent actions
        for i, [dc, ll_goal, policy] in enumerate(zip(self.ll_policy_ids, goal_list_ll, self.low_policies)):
            state_ll = np.concatenate([state['low_level_obs_' + dc], ll_goal])
            actions['low_level_action_' + dc] = np.clip(policy.select_action(state_ll), -1.0, 1.0)
            goal_list_dc[i].append(actions['low_level_action_' + dc][0]) # Append LS agent action to DC agent goal
        
        # DC agent actions
        for dc, dc_goal, policy in zip(self.ll_policy_ids, goal_list_dc, self.dc_policies):
            state_dc = np.concatenate([state['dc_obs_' + dc], dc_goal])
            actions['dc_action_' + dc] = policy.select_action(state_dc) # Discrete action space
        
        return actions
    
    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.high_policy.decay_action_std(action_std_decay_rate, min_action_std)
        for policy in self.low_policies:
            policy.decay_action_std(action_std_decay_rate, min_action_std)
    
    def update_policy(self, policy):
        """Helper function to update a single policy."""
        return policy.update()

    # Update with ThreadPoolExecutor
    def update(self):
        loss = []
        
        # List of all policies to be updated (high-level, low-level, DC)
        policies = [self.high_policy] + self.low_policies + self.dc_policies

        # Use ThreadPoolExecutor for multithreading
        with ThreadPoolExecutor(max_workers=len(policies)) as executor:
            # Submit each policy update task to the executor
            future_results = [executor.submit(self.update_policy, policy) for policy in policies]

            # Retrieve results from the futures as they complete
            for future in future_results:
                loss.append(future.result())

        return loss
    
    # Update with ProcessPoolExecutor
    # def update(self):
    #     loss = []

    #     # List of all policies (high-level, low-level, DC) that need to be updated
    #     policies = [self.high_policy] + self.low_policies + self.dc_policies

    #     # Use ProcessPoolExecutor for multiprocessing
    #     with ProcessPoolExecutor(max_workers=len(policies)) as executor:
    #         # Submit the update tasks to the process pool
    #         future_results = [executor.submit(self.update_policy, policy) for policy in policies]

    #         # Retrieve the results (loss values) from each process
    #         for future in future_results:
    #             loss.append(future.result())

    #     return loss
    
    # def update(self):
    #     loss = []
    #     loss.append(self.high_policy.update())
    #     for policy in self.low_policies:
    #         loss.append(policy.update())
    #     for policy in self.dc_policies:
    #         loss.append(policy.update())
    #     return loss
    
    def save(self, hl_checkpoint_path, ll_checkpoint_path, dc_checkpoint_path):
        self.high_policy.save(hl_checkpoint_path)
        for i, policy in enumerate(self.low_policies):
            path, ext = ll_checkpoint_path.rsplit('.', 1)
            policy.save(f"{path}_{i}.{ext}")
        for i, policy in enumerate(self.dc_policies):
            path, ext = dc_checkpoint_path.rsplit('.', 1)
            policy.save(f"{path}_{i}.{ext}")

    def load(self, hl_checkpoint_path, ll_checkpoint_path, dc_checkpoint_path):
        self.high_policy.load(hl_checkpoint_path)
        for i, policy in enumerate(self.low_policies):
            path, ext = ll_checkpoint_path.rsplit('.', 1)
            policy.load(f"{path}_{i}.{ext}")
        for i, policy in enumerate(self.dc_policies):
            path, ext = dc_checkpoint_path.rsplit('.', 1)
            policy.load(f"{path}_{i}.{ext}")

     