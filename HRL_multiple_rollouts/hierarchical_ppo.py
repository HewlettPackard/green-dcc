import os
import sys
import asyncio
import numpy as np
import torch

file_dir = os.path.dirname(__file__)
# append one level upper directory to python path
sys.path.append(file_dir + '/..')
# pylint: disable=C0301,C0303,C0103,C0209,C0116,C0413
import HRL_multiple_rollouts.ppo as ppo_class
from utils.utils_cf import generate_node_connections

class HierarchicalPPO:
    
    def __init__(self, num_ll_policies, 
                 obs_dim_hl, obs_dim_ll,
                 action_dim_hl, action_dim_ll,
                 goal_dim_ll,
                 hl_lr_actor, hl_lr_critic, ll_lr_actor, ll_lr_critic, 
                 hl_gamma, ll_gamma, hl_K_epochs, ll_K_epochs,
                 eps_clip,
                 hl_has_continuous_action_space = True, ll_has_continuous_action_space = True, 
                 action_std_init=0.6,  
                 high_policy_action_freq = 4,
                 ll_policy_ids = None):
        
        self.high_policy = ppo_class.PPO(obs_dim_hl, action_dim_hl,
                                        hl_lr_actor, hl_lr_critic, hl_gamma, hl_K_epochs, eps_clip, hl_has_continuous_action_space, action_std_init)
       
        self.low_policies = [ppo_class.PPO(state_dim + goal_dim, action_dim,
                                        ll_lr_actor, ll_lr_critic, ll_gamma, ll_K_epochs, eps_clip, ll_has_continuous_action_space, action_std_init)
                            for state_dim,goal_dim,action_dim in zip(obs_dim_ll, goal_dim_ll, action_dim_ll)]
        
        self.high_policy_action_freq = high_policy_action_freq  # number of selected actions before choosing the goal using the high policy
        self.action_counter = 0
        
        self.goal = np.random.uniform(-1, 1, action_dim_hl)  # a random goal to start with
        self.ll_policy_ids = ll_policy_ids
    
    def get_policy_params(self):
        """
        Returns the current parameters of the high-level and low-level policies.
        This method is used to send the policy parameters to the worker processes.
        """
        # Get parameters for the high-level policy
        high_level_policy_params = {k: v.clone() for k, v in self.high_policy.policy_old.state_dict().items()}

        # Get parameters for each low-level policy
        low_level_policy_params = []
        for policy in self.low_policies:
            params = {k: v.clone() for k, v in policy.policy_old.state_dict().items()}
            low_level_policy_params.append(params)

        # Return as a dictionary
        return {
            'high_policy': high_level_policy_params,
            'low_policies': low_level_policy_params
        }
    
    def load_policy_params(self, policy_params):
        """
        Loads the given policy parameters into the high-level and low-level policies.
        This method is used by the worker processes to update their local policies with the global parameters.
        """
        # Load parameters into the high-level policy
        self.high_policy.policy_old.load_state_dict(policy_params['high_policy'])

        # Load parameters into each low-level policy
        for policy, params in zip(self.low_policies, policy_params['low_policies']):
            policy.policy_old.load_state_dict(params)
            
    def select_action(self, state):
        actions = {}
        
        # high level actions
        if self.action_counter % self.high_policy_action_freq == 0:
            self.goal = self.high_policy.select_action(state['high_level_obs'])
        self.action_counter += 1
        
        actions['high_level_action'] = np.clip(self.goal, -1.0, 1.0)
        
        # mapping the actions from top level policy to the low level policy goal states
        goal_list = []
        for _, edges in generate_node_connections(N = [i for i in range(len(self.low_policies))], E = self.goal).items():
            goal_list.append([e[1] for e in edges])
        
        # generating the low level actions
        for i, j, policy in zip(self.ll_policy_ids, goal_list, self.low_policies):
            state_ll = np.concatenate([state['low_level_obs_' + i], j])
            actions['low_level_action_' + i] = np.clip(policy.select_action(state_ll), -1.0, 1.0)
             
        return actions
    
    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.high_policy.decay_action_std(action_std_decay_rate, min_action_std)
        for policy in self.low_policies:
            policy.decay_action_std(action_std_decay_rate, min_action_std)
            
    # def update(self):
    #     loss = []
    #     loss.append(self.high_policy.update())  # do I reduce the update frequency of the high policy? 
    #     # loss.append(self.high_policy.update()[0])
    #     # IndexError: too many indices for array: array is 0-dimensional, but 1 were indexed
    #     for policy in self.low_policies:
    #         loss.append(policy.update())
    #     return loss
    
    def update_with_experiences(self, aggregated_experiences):
        # Update high-level policy
        self.high_policy.buffer.states = [torch.tensor(s, dtype=torch.float32) for s in aggregated_experiences['high_policy']['states']]
        self.high_policy.buffer.actions = [torch.tensor(a, dtype=torch.float32) for a in aggregated_experiences['high_policy']['actions']]
        self.high_policy.buffer.logprobs = [torch.tensor(lp, dtype=torch.float32) for lp in aggregated_experiences['high_policy']['logprobs']]
        self.high_policy.buffer.rewards = [r for r in aggregated_experiences['high_policy']['rewards']]  # rewards can remain as a list of floats
        self.high_policy.buffer.state_values = [torch.tensor(v, dtype=torch.float32) for v in aggregated_experiences['high_policy']['state_values']]
        self.high_policy.buffer.is_terminals = [it for it in aggregated_experiences['high_policy']['is_terminals']]  # list of booleans

        high_policy_loss = self.high_policy.update()

        # Update low-level policies
        low_policy_losses = []
        for i, policy in enumerate(self.low_policies):
            experiences = aggregated_experiences['low_policies'][i]
            policy.buffer.states = [torch.tensor(s, dtype=torch.float32) for s in experiences['states']]
            policy.buffer.actions = [torch.tensor(a, dtype=torch.float32) for a in experiences['actions']]
            policy.buffer.logprobs = [torch.tensor(lp, dtype=torch.float32) for lp in experiences['logprobs']]
            policy.buffer.rewards = [r for r in experiences['rewards']]  # rewards can remain as a list of floats
            policy.buffer.state_values = [torch.tensor(v, dtype=torch.float32) for v in experiences['state_values']]
            policy.buffer.is_terminals = [it for it in experiences['is_terminals']]  # list of booleans
            loss = policy.update()
            low_policy_losses.append(loss)

        return [high_policy_loss] + low_policy_losses


    # async def async_update(self, policy):
    #     result = policy.update()  # Call the original method
    #     return result
    
    # async def update(self):
    #     loss = []

    #     # Create tasks for asynchronous updates
    #     high_policy_task = asyncio.create_task(self.async_update(self.high_policy))
    #     low_policy_tasks = [asyncio.create_task(self.async_update(policy)) for policy in self.low_policies]
    #     # Wait for all tasks to complete
    #     await asyncio.gather(high_policy_task, *low_policy_tasks)
    #     # Append results to loss list
    #     loss.append(high_policy_task.result())
    #     loss.extend(task.result() for task in low_policy_tasks)
        
    #     return loss
    
    def save(self, hl_checkpoint_path, ll_checkpoint_path):
        self.high_policy.save(hl_checkpoint_path)
        for i, policy in enumerate(self.low_policies):
            path, ext = ll_checkpoint_path.rsplit('.', 1)
            policy.save(f"{path}_{i}.{ext}")
            
    def load(self, hl_checkpoint_path, ll_checkpoint_path):
        self.high_policy.load(hl_checkpoint_path)
        for i, policy in enumerate(self.low_policies):
            path, ext = ll_checkpoint_path.rsplit('.', 1)
            policy.load(f"{path}_{i}.{ext}")
     