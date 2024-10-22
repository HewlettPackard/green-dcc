import os
import sys
file_dir = os.path.dirname(__file__)
# append one level upper directory to python path
sys.path.append(file_dir + '/..')
from envs.heirarchical_env_cont import HeirarchicalDCRL as HierarchicalDCRL # pylint: disable=E0401

# initialize HierarchicalDCRL with specific modifications from truly_heirarchical_env.py
class GreenDCC_Env(HierarchicalDCRL):

    def __init__(self,default_config=None):
        
        # update default config where initialize_queue_at_reset is set to False
        if default_config is not None:
            HierarchicalDCRL.__init__(self, config=default_config)
        else:
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
        # for action_space_l, dc in zip(self.action_space_ll, self.datacenter_ids):
            # assert action_space_l.contains(actions['low_level_action_' + dc]), f"Action: {actions['low_level_action_' + dc]} not in action space: {action_space_l}"
        
        # Move workload across DCs (high level policy)
        actions['high_level_action'] = self.transform_actions(actions['high_level_action'])
        original_workload, _ = self.safety_enforcement(actions['high_level_action'])  # overassigned_workload
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
        # infos['high_level_info'] = {}
        infos['high_level_info'] = {'hl_original_workload': original_workload}

        for dc in self.datacenter_ids:
            obs['low_level_obs_' + dc] = self.low_level_observations[dc]['agent_ls']
            # obs['dc_obs_' + dc] = self.low_level_observations[dc]['agent_dc']
            rewards['low_level_rewards_' + dc] = self.low_level_rewards[dc]['agent_ls']
            # rewards['dc_rewards_' + dc] = self.low_level_rewards[dc]['agent_dc']
            dones['low_level_done_' + dc] = done
            infos['low_level_info_' + dc] = {'CO2_footprint_per_step' : self.datacenters[dc].infos['agent_bat']['bat_CO2_footprint'],
                                             'bat_total_energy_with_battery_KWh' : self.datacenters[dc].infos['agent_bat']['bat_total_energy_with_battery_KWh'],
                                             'Carbon Intensity' : self.datacenters[dc].infos['agent_bat']['bat_avg_CI'],
                                             'External Temperature' : self.datacenters[dc].infos['agent_dc']['dc_exterior_ambient_temp'],
                                             'Spatial Shifted Workload' : self.datacenters[dc].infos['agent_ls']['ls_original_workload'],
                                             'Temporal Shifted Workload' : self.datacenters[dc].infos['agent_ls']['ls_shifted_workload'],
                                             'Water Consumption' : self.datacenters[dc].infos['agent_dc']['dc_water_usage'],
                                             'Queue Tasks' : self.datacenters[dc].infos['agent_ls']['ls_tasks_in_queue'],
                                             'Avg Age Task in Queue' : self.datacenters[dc].infos['agent_ls']['ls_average_task_age'],
                                             'ls_tasks_dropped': self.datacenters[dc].infos['agent_ls']['ls_tasks_dropped'],
                                             'ls_overdue_penalty':self.datacenters[dc].infos['agent_ls']['ls_overdue_penalty'],
                                             'Original Workload':original_workload[dc],
                                             }
            
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