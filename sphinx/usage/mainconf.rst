.. _mainconf_ref:

=========================
Main Configuration Files 
=========================

Edit the configuration files as needed to set up your desired benchmark parameters.

  - The configuration file for each simulated data center (number of cabinets, rows, HVAC configuration, etc.) can be found in the :code:`utils/dc_config_dcX.json` files where :code:`X=1,2,..` is the idetifier for each of the data centers.
  - Update the :code:`DEFAULT_CONFIG` in :code:`envs/hierarchical_env.py`.

Below is an example of the :code:`DEFAULT_CONFIG` in :code:`hierarchical_env.py` for a DCC with 3 data centers.


Example Configuration
-----------------------

.. code-block:: python 

     DEFAULT_CONFIG = {
        # DC1
        'config1': {
            'location': 'NY',
            'cintensity_file': 'NY_NG_&_avgCI.csv',
            'weather_file': 'USA_NY_New.York-LaGuardia.epw',
            'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv',
            'dc_config_file': 'dc_config_dc3.json',
            'datacenter_capacity_mw': 1.0,
            'timezone_shift': 0,
            'month': 7,
            'days_per_episode': 30,
            'partial_obs': True,
            'nonoverlapping_shared_obs_space': True
        },

        # DC2
        'config2': {
            'location': 'GA',
            'cintensity_file': 'GA_NG_&_avgCI.csv',
            'weather_file': 'USA_GA_Atlanta-Hartsfield-Jackson.epw',
            'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv',
            'dc_config_file': 'dc_config_dc2.json',
            'datacenter_capacity_mw': 1.0,
            'timezone_shift': 2,
            'month': 7,
            'days_per_episode': 30,
            'partial_obs': True,
            'nonoverlapping_shared_obs_space': True
        },

        # DC3
        'config3': {
            'location': 'CA',
            'cintensity_file': 'CA_NG_&_avgCI.csv',
            'weather_file': 'USA_CA_San.Jose-Mineta.epw',
            'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv',
            'dc_config_file': 'dc_config_dc1.json',
            'datacenter_capacity_mw': 1.0,
            'timezone_shift': 3,
            'month': 7,
            'days_per_episode': 30,
            'partial_obs': True,
            'nonoverlapping_shared_obs_space': True
        },
        
        # Number of transfers per step
        'num_transfers': 1,

        # List of active low-level agents
        'active_agents': ['agent_dc'],
    }

