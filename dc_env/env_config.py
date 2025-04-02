def get_default_config():
    return {
        # Agents active
        'agents': ['agent_ls', 'agent_dc', 'agent_bat'],

        # Datafiles
        'location': 'ny',
        'cintensity_file': 'NYIS_NG_&_avgCI.csv',
        'weather_file': 'USA_NY_New.York-Kennedy.epw',
        'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv',

        # Capacity (MW) of the datacenter
        'datacenter_capacity_mw': 1,

        # Timezone shift
        'timezone_shift': 0,

        # Days per simulated episode
        'days_per_episode': 7,

        # Maximum battery capacity
        'max_bat_cap_Mw': 2,

        # Data center configuration file
        'dc_config_file': 'dc_config.json',

        # Reward blending (individual/collaborative)
        'individual_reward_weight': 0.8,

        # Flexible load ratio of the total workload
        'flexible_load': 0.1,

        # Reward methods
        'ls_reward': 'default_ls_reward',
        'dc_reward': 'default_dc_reward',
        'bat_reward': 'default_bat_reward',

        # Offline evaluation flag
        'evaluation': False,
    }

class EnvConfig(dict):
    def __init__(self, raw_config):
        super().__init__(get_default_config())
        self.update(raw_config)
