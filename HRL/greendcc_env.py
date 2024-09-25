import os
import sys
file_dir = os.path.dirname(__file__)
# append one level upper directory to python path
sys.path.append(file_dir + '/..')
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
        
        self.goal_dimension_ll = [len(self.datacenter_ids)-1 for dc in self.datacenter_ids]  # also range in [-1,1]
        
    def reset(self, seed=None, options=None):
        super().reset(seed)
        obs = {}
        obs['high_level_obs'] = self.flat_obs
        for dc in self.datacenter_ids:
            obs['low_level_obs_' + dc] = self.low_level_observations[dc]['agent_ls']
        # observations are already in scaled range from internal evnironments. No need to scale them further
        return obs
    
    def seed(self, seed=None):  # pylint: disable=arguments-differ
        pass
    
    def step(self,actions: dict):
        
        # asssert agent actions are within action space
        assert self.action_space_hl.contains(actions['high_level_action']), f"Action: {actions['high_level_action']} not in action space: {self.action_space_hl}"
        for action_space_l, dc in zip(self.action_space_ll, self.datacenter_ids):
            assert action_space_l.contains(actions['low_level_action_' + dc]), f"Action: {actions['low_level_action_' + dc]} not in action space: {action_space_l}"
        
        # Move workload across DCs (high level policy)
        actions['high_level_action'] = self.transform_actions(actions['high_level_action'])
        _ = self.safety_enforcement(actions['high_level_action'])  # overassigned_workload
        # Move workload within DCs (low level policy)
        low_level_actions = {dc: {'agent_ls': actions['low_level_action_' + dc]} for dc in self.datacenter_ids}
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
            rewards['low_level_rewards_' + dc] = self.low_level_rewards[dc]['agent_ls']
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