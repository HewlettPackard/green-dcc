import os
import sys
file_dir = os.path.dirname(__file__)
# append one level upper directory to python path
sys.path.append(file_dir + '/..')
import numpy as np
from envs.heirarchical_env_cont import HeirarchicalDCRL as HierarchicalDCRL # pylint: disable=E0401

# initialize HierarchicalDCRL with specific modifications from truly_heirarchical_env.py
class GreenDCC_Env(HierarchicalDCRL):

    def __init__(self,):
        HierarchicalDCRL.__init__(self)
        
        # top level agent obs space is defined inside heirarchical_env_cont.py around L162
        # top level agent action space is defined inside heirarchical_env_cont.py around L168
        self.observation_space_hl = self.observation_space
        self.action_space_hl = self.action_space
        
        self.observation_space_ll = [self.datacenters[dc].ls_env.observation_space for dc in self.datacenter_ids]
        self.action_space_ll = [self.datacenters[dc].ls_env.action_space for dc in self.datacenter_ids]
        
        self.observation_space_dc = [self.datacenters[dc].dc_env.observation_space for dc in self.datacenter_ids]
        self.action_space_dc = [self.datacenters[dc].dc_env.action_space for dc in self.datacenter_ids]
        
        self.goal_dimension_ll = [len(self.datacenter_ids)-1 for dc in self.datacenter_ids]  # The transfered workload from the top level agent (hl)
        self.goal_dimension_dc = [len(self.datacenter_ids)-1 + 1 for dc in self.datacenter_ids]  # The transfered workload from the top level agent (hl) and the shifted workload from the low level agent (ll)
        
    def reset(self, seed=None, options=None):
        super().reset(seed)
        obs = {}
        obs['high_level_obs'] = self.flat_obs
        for dc in self.datacenter_ids:
            obs['low_level_obs_' + dc] = self.low_level_observations[dc]['agent_ls']
            obs['dc_obs_' + dc] = self.low_level_observations[dc]['agent_dc']
        # observations are already in scaled range from internal evnironments. No need to scale them further
        return obs
    
    def seed(self, seed=None):  # pylint: disable=arguments-differ
        pass
    
    def step(self, actions: dict):
        
        # asssert agent actions are within action space
        assert self.action_space_hl.contains(actions['high_level_action']), f"Action: {actions['high_level_action']} not in action space: {self.action_space_hl}"
        for action_space_l, action_space_d, dc in zip(self.action_space_ll, self.action_space_dc, self.datacenter_ids):
            assert action_space_l.contains(actions['low_level_action_' + dc]), f"Action: {actions['low_level_action_' + dc]} not in action space: {action_space_l}"
            assert action_space_d.contains(actions['dc_action_' + dc]), f"Action: {actions['dc_action_' + dc]} not in action space: {action_space_d}"
        
        # Move workload across DCs (high level policy)
        actions['high_level_action'] = self.transform_actions(actions['high_level_action'])
        _ = self.safety_enforcement(actions['high_level_action'])  # overassigned_workload
        # Move workload within DCs (low level policy)
        low_level_actions = {dc: {'agent_ls': np.clip(actions['low_level_action_' + dc], self.action_space_ll[0].low, self.action_space_ll[0].high),
                                  'agent_dc': actions['dc_action_' + dc]
                                  } 
                             for dc in self.datacenter_ids}
        done = self.low_level_step(low_level_actions)
        
        obs = {}
        rewards = {}
        dones = {}
        infos = {}
        
        obs['high_level_obs'] = self.flat_obs
        rewards['high_level_rewards'] = self.calc_reward()
        dones['high_level_done'] = done
        infos['high_level_info'] = {}
        
        for dc in self.datacenter_ids:
            obs['low_level_obs_' + dc] = self.low_level_observations[dc]['agent_ls']
            obs['dc_obs_' + dc] = self.low_level_observations[dc]['agent_dc']
            rewards['low_level_rewards_' + dc] = self.low_level_rewards[dc]['agent_ls']
            rewards['dc_rewards_' + dc] = self.low_level_rewards[dc]['agent_dc']
            dones['low_level_done_' + dc] = done
            infos['low_level_info_' + dc] = {'CO2_footprint_per_step' : self.metrics[dc]['bat_CO2_footprint'],
                                             'bat_total_energy_with_battery_KWh' : self.metrics[dc]['bat_total_energy_with_battery_KWh'],
                                             'ls_tasks_dropped': self.metrics[dc]['ls_tasks_dropped'],}
            
        return obs, rewards, dones, infos
        
if __name__ == "__main__":
    env = GreenDCC_Env()
    print("Environment created")
    print("Observation Space HL: ", env.observation_space_hl)
    print("Action Space HL: ", env.action_space_hl)
    print("Observation Space LL: ", env.observation_space_ll)
    print("Action Space LL: ", env.action_space_ll)
    print(env.observation_space_hl.shape[0])
    print(env.action_space_hl.shape[0])
    print(env.observation_space_ll[0].shape[0])
    print(env.action_space_ll[0].shape[0])
    
    obs = env.reset()
    
    print("Reset done")
    print("Observation: ", obs)
    
    actions = {}
    actions['high_level_action'] = env.action_space_hl.sample()
    for dc in env.datacenter_ids:
        actions['low_level_action_' + dc] = env.action_space_ll[0].sample()
        actions['dc_action_' + dc] = env.action_space_dc[0].sample()
    
    print("Actions: ", actions)
    obs, rewards, dones, infos = env.step(actions)
    print("Step done")
    print("Observation: ", obs)
    print("Rewards: ", rewards)
    print("Dones: ", dones)
    print("Infos: ", infos)
    print("Observation HL: ", obs['high_level_obs'])
    print("Observation LL: ", obs['low_level_obs_' + env.datacenter_ids[0]])
    print("Rewards HL: ", rewards['high_level_rewards'])
    print("Rewards LL: ", rewards['low_level_rewards_' + env.datacenter_ids[0]])
    print("CO2_footprint_per_step: ", infos['low_level_info_' + env.datacenter_ids[0]]['CO2_footprint_per_step'])
    print("bat_total_energy_with_battery_KWh: ", infos['low_level_info_' + env.datacenter_ids[0]]['bat_total_energy_with_battery_KWh'])
    print("ls_tasks_dropped: ", infos['low_level_info_' + env.datacenter_ids[0]]['ls_tasks_dropped'])
    
    env.close()
    print("Environment closed")
    
    