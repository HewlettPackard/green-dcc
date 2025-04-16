def get_default_config():
    return {
        # Agents active
        'agents': [],

        # Datafiles
        'location': 'ny',

        # Timezone shift
        'timezone_shift': 0,

        # Maximum battery capacity
        'max_bat_cap_Mw': 2,

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
