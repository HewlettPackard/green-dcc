# File where the rewards are defined
import numpy as np

bat_dcload = []
# bat_footprint is a dictionary that stores the CO2 footprint of each data center
# The key is the data center location and the value is a list of CO2 footprints
bat_footprint = {}
norm_cfp_values = {'ny': {'mean': 53174, 'std': 14873},
                   'ca': {'mean': 49973, 'std': 17429},
                   'az': {'mean': 142121, 'std': 48681},
}

norm_energy_values = {'ny': {'mean': 173, 'std': 45},
                      'ca': {'mean': 170, 'std': 49},
                      'az': {'mean': 234, 'std': 78},
    }
def default_ls_reward(params: dict) -> float:
    """
    Calculates a reward value based on normalized load shifting.

    Args:
        params (dict): Dictionary containing parameters:
            norm_load_left (float): Normalized load left.
            out_of_time (bool): Indicator (alarm) whether the agent is in the last hour of the day.
            penalty (float): Penalty value.

    Returns:
        float: Reward value.
    """
    location = params['location']
    total_energy = params['bat_total_energy_with_battery_KWh']
    norm_total_energy = (total_energy - norm_energy_values[location]['mean']) / norm_energy_values[location]['std']
    
    norm_ci = params['norm_CI']
    # if location not in bat_footprint:
        # bat_footprint[location] = []
    # bat_footprint[location].append(total_energy)
    
    footprint_reward = -1.0 * (norm_ci * norm_total_energy / 0.50)  # Mean and std reward. Negate to maximize reward and minimize energy consumption
    # footprint_reward = -1.0 * (total_CFP - norm_values[location]['mean']) / norm_values[location]['std']  # Mean and std reward. Negate to maximize reward and minimize energy consumption
    
    # Overdue Tasks Penalty (scaled)
    overdue_penalty_scale = 5.0  # Adjust this scaling factor as needed
    overdue_penalty_bias = 1.0
    # tasks_overdue_penalty = -overdue_penalty_scale * np.log(params['ls_overdue_penalty'] + 1) # +1 to avoid log(0) and be always negative
    tasks_overdue_penalty = -overdue_penalty_scale * np.sqrt(params['ls_overdue_penalty']) + overdue_penalty_bias # To have a +1 if the number of overdue tasks is 0, and a negative value otherwise
    # Oldest Task Age Penalty
    age_penalty_scale = 0.5  # Adjust this scaling factor as needed
    tasks_age_penalty = -age_penalty_scale * params['ls_oldest_task_age']  # Assume normalized between 0 and 1

    # Total Reward
    total_reward = footprint_reward + tasks_overdue_penalty + tasks_age_penalty

    return total_reward


def default_dc_reward(params: dict) -> float:
    """
    Calculates a reward value based on the data center's total ITE Load and CT Cooling load.

    Args:
        params (dict): Dictionary containing parameters:
            data_center_total_ITE_Load (float): Total ITE Load of the data center.
            CT_Cooling_load (float): CT Cooling load of the data center.
            energy_lb (float): Lower bound of the energy.
            energy_ub (float): Upper bound of the energy.

    Returns:
        float: Reward value.
    """
    data_center_total_ITE_Load = params['dc_ITE_total_power_kW']
    CT_Cooling_load = params['dc_HVAC_total_power_kW']
    energy_lb,  energy_ub = params['dc_power_lb_kW'], params['dc_power_ub_kW']
    
    return - 1.0 * ((data_center_total_ITE_Load + CT_Cooling_load) - energy_lb) / (energy_ub - energy_lb)


def default_bat_reward(params: dict) -> float:
    """
    Calculates a reward value based on the battery usage.

    Args:
        params (dict): Dictionary containing parameters:
            total_energy_with_battery (float): Total energy with battery.
            norm_CI (float): Normalized Carbon Intensity.
            dcload_min (float): Minimum DC load.
            dcload_max (float): Maximum DC load.

    Returns:
        float: Reward value.
    """
    total_energy_with_battery = params['bat_total_energy_with_battery_KWh']
    norm_CI = params['norm_CI']
    dcload_min = params['bat_dcload_min']
    dcload_max = params['bat_dcload_max']
    
    norm_net_dc_load = (total_energy_with_battery - dcload_min) / (dcload_max - dcload_min)
    rew_footprint = -1.0 * norm_CI * norm_net_dc_load #Added scalar to line up with dc reward

    return rew_footprint


def custom_agent_reward(params: dict) -> float:
    """
    A template for creating a custom agent reward function.

    Args:
        params (dict): Dictionary containing custom parameters for reward calculation.

    Returns:
        float: Custom reward value. Currently returns 0.0 as a placeholder.
    """
    # read reward input parameters from dict object
    # custom reward calculations 
    custom_reward = 0.0 # update with custom reward shaping 
    return custom_reward

# Example of ToU reward based on energy usage and price of electricity
# ToU reward is based on the ToU (Time of Use) of the agent, which is the amount of the energy time
# the agent spends on the grid times the price of the electricity.
# This example suppose that inside the params there are the following keys:
#   - 'energy_usage': the energy usage of the agent
#   - 'hour': the hour of the day
def tou_reward(params: dict) -> float:
    """
    Calculates a reward value based on the Time of Use (ToU) of energy.

    Args:
        params (dict): Dictionary containing parameters:
            energy_usage (float): The energy usage of the agent.
            hour (int): The current hour of the day (24-hour format).

    Returns:
        float: Reward value.
    """
    
    # ToU dict: {Hour, price}
    tou = {0: 0.25,
           1: 0.25,
           2: 0.25,
           3: 0.25,
           4: 0.25,
           5: 0.25,
           6: 0.41,
           7: 0.41,
           8: 0.41,
           9: 0.41,
           10: 0.41,
           11: 0.30,
           12: 0.30,
           13: 0.30,
           14: 0.30,
           15: 0.30,
           16: 0.27,
           17: 0.27,
           18: 0.27,
           19: 0.27,
           20: 0.27,
           21: 0.27,
           22: 0.25,
           23: 0.25,
           }
    
    # Obtain the price of electricity at the current hour
    current_price = tou[params['hour']]
    # Obtain the energy usage
    energy_usage = params['bat_total_energy_with_battery_KWh']
    
    # The reward is negative as the agent's objective is to minimize energy cost
    tou_reward = -1.0 * energy_usage * current_price

    return tou_reward


def renewable_energy_reward(params: dict) -> float:
    """
    Calculates a reward value based on the usage of renewable energy sources.

    Args:
        params (dict): Dictionary containing parameters:
            renewable_energy_ratio (float): Ratio of energy coming from renewable sources.
            total_energy_consumption (float): Total energy consumption of the data center.

    Returns:
        float: Reward value.
    """
    assert params.get('renewable_energy_ratio') is not None, 'renewable_energy_ratio is not defined. This parameter should be included using some external dataset and added to the reward_info dictionary'
    renewable_energy_ratio = params['renewable_energy_ratio'] # This parameter should be included using some external dataset
    total_energy_consumption = params['bat_total_energy_with_battery_KWh']
    factor = 1.0 # factor to scale the weight of the renewable energy usage

    # Reward = maximize renewable energy usage - minimize total energy consumption
    reward = factor * renewable_energy_ratio  -1.0 * total_energy_consumption
    return reward


def energy_efficiency_reward(params: dict) -> float:
    """
    Calculates a reward value based on energy efficiency.

    Args:
        params (dict): Dictionary containing parameters:
            ITE_load (float): The amount of energy spent on computation (useful work).
            total_energy_consumption (float): Total energy consumption of the data center.

    Returns:
        float: Reward value.
    """
    it_equipment_power = params['dc_ITE_total_power_kW']  
    total_power_consumption = params['dc_total_power_kW']  
    
    reward = it_equipment_power / total_power_consumption
    return reward


def energy_PUE_reward(params: dict) -> float:
    """
    Calculates a reward value based on Power Usage Effectiveness (PUE).

    Args:
        params (dict): Dictionary containing parameters:
            total_energy_consumption (float): Total energy consumption of the data center.
            it_equipment_energy (float): Energy consumed by the IT equipment.

    Returns:
        float: Reward value.
    """
    total_power_consumption = params['dc_total_power_kW']  
    it_equipment_power = params['dc_ITE_total_power_kW']  
    
    # Calculate PUE
    pue = total_power_consumption / it_equipment_power if it_equipment_power != 0 else float('inf')
    
    # We aim to get PUE as close to 1 as possible, hence we take the absolute difference between PUE and 1
    # We use a negative sign since RL seeks to maximize reward, but we want to minimize PUE
    reward = -abs(pue - 1)
    
    return reward


def temperature_efficiency_reward(params: dict) -> float:
    """
    Calculates a reward value based on the efficiency of cooling in the data center.

    Args:
        params (dict): Dictionary containing parameters:
            current_temperature (float): Current temperature in the data center.
            optimal_temperature_range (tuple): Tuple containing the minimum and maximum optimal temperatures for the data center.

    Returns:
        float: Reward value.
    """
    assert params.get('optimal_temperature_range') is not None, 'optimal_temperature_range is not defined. This parameter should be added to the reward_info dictionary'
    current_temperature = params['dc_int_temperature'] 
    optimal_temperature_range = params['optimal_temperature_range']
    min_temp, max_temp = optimal_temperature_range
    
    if min_temp <= current_temperature <= max_temp:
        reward = 1.0
    else:
        if current_temperature < min_temp:
            reward = -abs(current_temperature - min_temp)
        else:
            reward = -abs(current_temperature - max_temp)
    return reward

def water_usage_efficiency_reward(params: dict) -> float:
    """
    Calculates a reward value based on the efficiency of water usage in the data center.
    
    A lower value of water usage results in a higher reward, promoting sustainability
    and efficiency in water consumption.

    Args:
        params (dict): Dictionary containing parameters:
            dc_water_usage (float): The amount of water used by the data center in a given period.

    Returns:
        float: Reward value. The reward is higher for lower values of water usage, 
        promoting reduced water consumption.
    """
    dc_water_usage = params['dc_water_usage']
    
    # Calculate the reward. This is a simple inverse relationship; many other functions could be applied.
    # Adjust the scalar as needed to fit the scale of your rewards or to emphasize the importance of water savings.
    reward = -0.01 * dc_water_usage
    
    return reward

# Other reward methods can be added here.

REWARD_METHOD_MAP = {
    'default_dc_reward' : default_dc_reward,
    'default_bat_reward': default_bat_reward,
    'default_ls_reward' : default_ls_reward,
    # Add custom reward methods here
    'custom_agent_reward' : custom_agent_reward,
    'tou_reward' : tou_reward,
    'renewable_energy_reward' : renewable_energy_reward,
    'energy_efficiency_reward' : energy_efficiency_reward,
    'energy_PUE_reward' : energy_PUE_reward,
    'temperature_efficiency_reward' : temperature_efficiency_reward,
    'water_usage_efficiency_reward' : water_usage_efficiency_reward,
}

def get_reward_method(reward_method : str = 'default_dc_reward'):
    """
    Maps the string identifier to the reward function

    Args:
        reward_method (string): Identifier for the reward function.

    Returns:
        function: Reward function.
    """
    assert reward_method in REWARD_METHOD_MAP.keys(), f"Specified Reward Method {reward_method} not in REWARD_METHOD_MAP"
    
    return REWARD_METHOD_MAP[reward_method]

