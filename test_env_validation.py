from gymnasium.utils.env_checker import check_env
from dc_env.dc_scheduling_env import TaskSchedulingEnv
from simulation.datacenter_cluster_manager import DatacenterClusterManager
import pandas as pd
import datetime

def make_env():
    simulation_year = 2023
    simulated_month = 8
    init_day = 1
    init_hour = 5
    init_minute = 0

    start_time = datetime.datetime(simulation_year, simulated_month, init_day, init_hour, init_minute, tzinfo=datetime.timezone.utc)
    end_time = start_time + datetime.timedelta(days=1)
    start_time = pd.Timestamp(start_time)
    end_time = pd.Timestamp(end_time)

    datacenter_configs = [
        {
            'location': 'US-NY-NYIS', 'dc_id': 1, 'agents': [], 'timezone_shift': -5,
            'dc_config_file': 'dc_config.json', 'weather_file': None, 'cintensity_file': None,
            'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv', 'month': simulated_month,
            'datacenter_capacity_mw': 1.5, 'max_bat_cap_Mw': 3.0, 'days_per_episode': 30,
            'network_cost_per_gb': 0.08, 'total_cpus': 3000, 'total_gpus': 600,
            'total_mem': 6000, 'population_weight': 0.25,
        }
    ]

    tasks_file_path = "data/workload/alibaba_2020_dataset/result_df_full_year_2020.pkl"

    cluster_manager = DatacenterClusterManager(
        config_list=datacenter_configs,
        simulation_year=simulation_year,
        init_day=init_day,
        init_hour=init_hour,
        strategy="priority_order",
        tasks_file_path=tasks_file_path
    )

    env = TaskSchedulingEnv(
        cluster_manager=cluster_manager,
        start_time=start_time,
        end_time=end_time,
        carbon_price_per_kg=0.1
    )
    return env

# === Validate ===
if __name__ == "__main__":
    env = make_env()
    check_env(env, skip_render_check=True)
    print("Environment passed Gymnasium validation.")
