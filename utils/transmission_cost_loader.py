import pandas as pd
import os

_TRANSMISSION_COST_CACHE = {}

def load_transmission_matrix(cloud_provider: str):
    """
    Loads and caches the transmission matrix CSV for a given cloud provider.
    """
    global _TRANSMISSION_COST_CACHE
    cloud_provider = cloud_provider.lower()

    if cloud_provider in _TRANSMISSION_COST_CACHE:
        return _TRANSMISSION_COST_CACHE[cloud_provider]

    base_path = "data/network_cost"
    fname = f"{cloud_provider}_transmission_cost_matrix.csv"
    fpath = os.path.join(base_path, fname)

    df = pd.read_csv(fpath, index_col=0)
    df = df.fillna(0.0)
    _TRANSMISSION_COST_CACHE[cloud_provider] = df
    return df
