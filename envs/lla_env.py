from tqdm import tqdm
from ray.rllib.env import MultiAgentEnv

from envs.heirarchical_env_cont import HeirarchicalDCRL, DEFAULT_CONFIG
from tensorboardX import SummaryWriter

class LLADCRL(HeirarchicalDCRL, MultiAgentEnv):

    def __init__(self, config):
        # super(LLADCRL, self).__init__(config)
        HeirarchicalDCRL.__init__(self, config)
        MultiAgentEnv.__init__(self)
        self.writer = SummaryWriter("logs_single")
        self.global_step = 0


    def reset(self, seed=None, options=None):
        super().reset(seed)

        obs = {}
        for dc in self.datacenter_ids:
            obs[dc + '_ls_policy'] = self.low_level_observations[dc]['agent_ls']
        return obs, {}
    
    def step(self, actions: dict):
        # Move workload across DCs (high level policy)
        # Move workload within DCs
        low_level_actions = {dc: {'agent_ls': actions[dc + '_ls_policy']} for dc in self.datacenter_ids}
        done = self.low_level_step(low_level_actions)
        
        # Get observations for high-level agent
        # if not done:
        #     self.heir_obs = {}
        #     for dc in self.datacenter_ids:
        #         self.heir_obs[dc] = self.get_dc_variables(dc)

        # Prepare high-level obs and rewards
        obs = {}
        rewards = {}
        # rewards['high_level_policy'] = self.calc_reward()

        # Low-level obs and rewards
        for dc in self.datacenter_ids:
            obs[dc + '_ls_policy'] = self.low_level_observations[dc]['agent_ls']
            rewards[dc + '_ls_policy'] = self.low_level_rewards[dc]['agent_ls']

        # Infinite horizon env so it is never terminated, only truncated
        terminated = {"__all__": False}
        truncated = {"__all__": done}
        
        if done:
            totalfp = 0
            for dc in self.datacenter_ids:
                totalfp += sum(self.metrics[dc]['bat_CO2_footprint'])
            # print("Total CO2 footprint: ", totalfp)
            print(f'The total CO2 footprint is {totalfp}')

            # Log the scalar totalfp to TensorBoard
            self.writer.add_scalar("Total CO2 footprint", totalfp, self.global_step)
            self.writer.flush()
            self.global_step += 1  # Increment the step counter

        return obs, rewards, terminated, truncated, {}
    
if __name__ == '__main__':
    env = LLADCRL(DEFAULT_CONFIG)

    done = False
    obs, _ = env.reset()
    
    with tqdm(total=env._max_episode_steps) as pbar:
        while not done:
            actions = {}
            for dc in env.datacenter_ids:
                actions[dc + '_ls_policy'] = env.datacenters['DC1'].ls_env.action_space.sample()

            obs, rewards, terminated, truncated, info = env.step(actions)
            done = truncated['__all__']

            pbar.update(1)