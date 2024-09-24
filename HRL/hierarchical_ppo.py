import os
import sys
import numpy as np

file_dir = os.path.dirname(__file__)
# append one level upper directory to python path
sys.path.append(file_dir + '/..')
# pylint: disable=C0301,C0303,C0103,C0209,C0116,C0413
import HRL.ppo as ppo_class
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
        
    def select_action(self,state):
        
        actions = {}
        
        # high level actions
        if self.action_counter % self.high_policy_action_freq == 0:
            self.goal = self.high_policy.select_action(state['high_level_obs'])
        self.action_counter += 1
        
        actions['high_level_action'] = np.clip(self.goal,-1.0,1.0)
        
        # mapping the actions from top level policy to the low level policy goal states
        goal_list = []
        for _, edges in generate_node_connections(N = [i for i in range(len(self.low_policies))], E = self.goal).items():
            goal_list.append([e[1] for e in edges])
        
        # generating the low level actions
        for i,j,policy in zip(self.ll_policy_ids,goal_list,self.low_policies):
            state_ll = np.concatenate([state['low_level_obs_' + i], j])
            actions['low_level_action_' + i] = np.clip(policy.select_action(state_ll),-1.0,1.0)
            
        return actions
    
    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.high_policy.decay_action_std(action_std_decay_rate, min_action_std)
        for policy in self.low_policies:
            policy.decay_action_std(action_std_decay_rate, min_action_std)
            
    def update(self):
        loss = []
        loss.append(self.high_policy.update())  # do I reduce the update frequency of the high policy? 
        # loss.append(self.high_policy.update()[0])
        # IndexError: too many indices for array: array is 0-dimensional, but 1 were indexed
        for policy in self.low_policies:
            loss.append(policy.update())
        return loss
    
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
     