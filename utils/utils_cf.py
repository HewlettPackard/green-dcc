import os
import numpy as np
import pandas as pd

import os

file_path = os.path.abspath(__file__)
PATH = os.path.split(os.path.dirname(file_path))[0]


def obtain_paths(location):
    """Obtain the correct name for the data files

    Args:
        location (string): Location identifier

    Raises:
        ValueError: If location identifier is not defined

    Returns:
        List[string]: Naming for the data files
    """
    if 'az' in location.lower():
        return ['AZ', 'USA_AZ_Phoenix-Sky.Harbor.epw']
    elif 'ca' in location.lower():
        return ['CA', 'USA_CA_San.Jose-Mineta.epw']
    elif 'ga' in location.lower():
        return ['GA', 'USA_GA_Atlanta-Hartsfield-Jackson.epw']
    elif 'il' in location.lower():
        return ['IL', 'USA_IL_Chicago.OHare.epw']
    elif 'ny' in location.lower():
        return ['NY', 'USA_NY_New.York-LaGuardia.epw']
    elif 'tx' in location.lower():
        return ['TX', 'USA_TX_Dallas-Fort.Worth.epw']
    elif 'va' in location.lower():
        return ['VA', 'USA_VA_Leesburg.Exec.epw']
    elif "wa" in location.lower():
        return ['WA', 'USA_WA_Seattle-Tacoma.epw']
    else:
        raise ValueError(f"Location not found, please define the location {location} on obtain_paths() function in utils_cf.py")

def get_energy_variables(state):
    """Obtain energy variables from the energy observation

    Args:
        state (List[float]): agent_dc observation

    Returns:
        List[float]: Subset of the agent_dc observation
    """
    energy_vars = np.hstack((state[4:7],(state[7]+state[8])/2))
    return energy_vars

import json
# Function to get the initial index of the day of a given month from a time-stamped dataset
def get_init_day(weather_file, start_month=0):
    """
    Obtain the initial day of the year to start the episode.

    Args:
        weather_file (str): Path to the weather JSON file.
        start_month (int, optional): Starting month (0=Jan, 11=Dec). Defaults to 0.

    Returns:
        int: Day of the year corresponding to the first day of the month.
    """
    assert 0 <= start_month <= 11, "start_month should be between 0 and 11 (inclusive, 0=January, 11=December)."

    # Load the JSON weather data
    with open(weather_file, 'r') as f:
        weather_data = json.load(f)

    # Convert timestamps to pandas datetime format
    timestamps = pd.to_datetime(weather_data["hourly"]["time"])

    # Extract the month for each timestamp
    months = timestamps.month

    # Find the first occurrence of the desired month
    first_index = (months == start_month + 1).argmax()

    # Convert index to day (each day has 24 hours)
    init_day = first_index // 24

    return init_day