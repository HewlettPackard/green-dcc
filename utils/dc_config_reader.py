"""
This file is used to read the data center configuration from  user inputs provided inside dc_config.json. It also performs some auxiliary steps to calculate the server power specifications based on the given parameters.
"""
import os
import math
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

class DC_Config:
    def __init__(self, dc_config_file='dc_config.json', total_cores=0, total_gpus=0, total_mem=0, datacenter_capacity_mw=0):
        """
        Initializes a new instance of the DC_Config class, loading configuration
        data from the specified JSON configuration file.

        Args:
            dc_config_file (str): The name of the data center configuration JSON file 
                                  (which should live in ../configs).
            datacenter_capacity_mw (float): The maximum compute power capacity of the datacenter in MW.
        """
        self.config_path = os.path.abspath(dc_config_file)

        # Keep the capacity parameter
        self.datacenter_capacity_mw = datacenter_capacity_mw
        
        self.total_cores = total_cores
        self.total_gpus = total_gpus
        self.total_mem = total_mem
        
        self.CORES_PER_SERVER = 64
        self.MAX_W_PER_CORE = 6  # Watts per core
        self.GPUS_PER_SERVER = 4
        self.MAX_SERVERS_PER_RACK = 25
        
        # Load the JSON data from the configuration file
        self.config_data = self._load_config()

        # Set up configuration parameters
        self._setup_config()
        
    def _load_config(self):
        """
        Loads the data center configuration from the specified JSON file.

        Returns:
            dict: A dictionary containing the loaded configuration data.
        """
        with open(self.config_path, 'r') as file:
            return json.load(file)
    
    def _setup_config(self):
        dc = self.config_data['data_center_configuration']
        servers = self.config_data['server_characteristics']
        hvac = self.config_data['hvac_configuration']

        # === Compute total number of servers needed ===
        self.total_servers_cpu = math.ceil(self.total_cores / self.CORES_PER_SERVER)
        self.total_servers_gpu = math.ceil(self.total_gpus / self.GPUS_PER_SERVER)
        self.total_servers = max(self.total_servers_cpu, self.total_servers_gpu)

        # === Racks based on max 25 servers per rack ===
        self.NUM_RACKS = math.ceil(self.total_servers / self.MAX_SERVERS_PER_RACK)
        self.CPUS_PER_RACK = math.ceil(self.total_cores / self.NUM_RACKS)
        self.GPUS_PER_RACK = math.ceil(self.total_gpus / self.NUM_RACKS)

        print(f"[INFO] Using {self.NUM_RACKS} racks with max {self.MAX_SERVERS_PER_RACK} servers per rack")
        print(f"[INFO] CPUs per rack: {self.CPUS_PER_RACK}, GPUs per rack: {self.GPUS_PER_RACK}")

        # === Use average rack-level values ===
        avg_supply_temp = sum(dc['RACK_SUPPLY_APPROACH_TEMP_LIST']) / len(dc['RACK_SUPPLY_APPROACH_TEMP_LIST'])
        avg_return_temp = sum(dc['RACK_RETURN_APPROACH_TEMP_LIST']) / len(dc['RACK_RETURN_APPROACH_TEMP_LIST'])

        self.RACK_SUPPLY_APPROACH_TEMP_LIST = [avg_supply_temp] * self.NUM_RACKS
        self.RACK_RETURN_APPROACH_TEMP_LIST = [avg_return_temp] * self.NUM_RACKS

        # avg_cpu_power = self._avg_pair_list(self.config_data['server_characteristics']['DEFAULT_SERVER_POWER_CHARACTERISTICS'])
        # The average CPU power is calculated based on the 20 Watts per core and the number of cores per server
        avg_cpu_power = [self.MAX_W_PER_CORE*self.CORES_PER_SERVER, 20]  # [full_load_pwr, idle_pwr]
        avg_gpu_power = self._avg_pair_list(self.config_data['server_characteristics']['DEFAULT_GPU_POWER_CHARACTERISTICS'])

        self.DEFAULT_SERVER_POWER_CHARACTERISTICS = [avg_cpu_power] * self.NUM_RACKS
        self.DEFAULT_GPU_POWER_CHARACTERISTICS = [avg_gpu_power] * self.NUM_RACKS

        # === Build power config per rack ===
        def make_cpu_cfg(spec):
            num_servers = math.ceil(self.CPUS_PER_RACK / self.CORES_PER_SERVER)
            return [{'full_load_pwr': spec[0], 'idle_pwr': spec[1]} for _ in range(num_servers)]
            # return [{'full_load_pwr': spec[0], 'idle_pwr': spec[1]} for _ in range(self.CPUS_PER_RACK)]

        def make_gpu_cfg(spec):
            return [{'full_load_pwr': spec[0], 'idle_pwr': spec[1]} for _ in range(self.GPUS_PER_RACK)]

        with ThreadPoolExecutor() as executor:
            cpu_futures = [executor.submit(make_cpu_cfg, spec) for spec in self.DEFAULT_SERVER_POWER_CHARACTERISTICS]
            gpu_futures = [executor.submit(make_gpu_cfg, spec) for spec in self.DEFAULT_GPU_POWER_CHARACTERISTICS]
            self.RACK_CPU_CONFIG = [f.result() for f in as_completed(cpu_futures)]
            self.RACK_GPU_CONFIG = [f.result() for f in as_completed(gpu_futures)]
        
        # Compute actual MAX_W_PER_RACK based on final rack contents
        self.MAX_W_PER_RACK = max(
            sum(cpu['full_load_pwr'] for cpu in rack_cpu) + 
            sum(gpu['full_load_pwr'] for gpu in rack_gpu)
            for rack_cpu, rack_gpu in zip(self.RACK_CPU_CONFIG, self.RACK_GPU_CONFIG)
        )
        
        print("\n[INFO] --- Rack Device Summary ---")
        for i, (rack_cpu, rack_gpu) in enumerate(zip(self.RACK_CPU_CONFIG, self.RACK_GPU_CONFIG)):
            num_servers = len(rack_cpu)
            num_gpus = len(rack_gpu)

            total_cores = num_servers * self.CORES_PER_SERVER
            cpu_power = sum(cpu['full_load_pwr'] for cpu in rack_cpu)
            gpu_power = sum(gpu['full_load_pwr'] for gpu in rack_gpu)

            print(f"Rack {i+1:02d}: {num_servers} CPU servers × {self.CORES_PER_SERVER} cores → {total_cores} cores, "
                f"{num_gpus} GPUs → Total theory Power ≈ {cpu_power + gpu_power:.1f} W")

        # A default value of HP_PROLIANT server for standalone testing
        self.HP_PROLIANT = servers['HP_PROLIANT']

        # A default value of HP_PROLIANT server for standalone testing
        self.NVIDIA_V100 = servers['NVIDIA_V100']

        # Serve/cpu parameters; Obtained from [3]
        self.CPU_POWER_RATIO_LB = servers['CPU_POWER_RATIO_LB']
        self.CPU_POWER_RATIO_UB = servers['CPU_POWER_RATIO_UB']
        self.IT_FAN_AIRFLOW_RATIO_LB = servers['IT_FAN_AIRFLOW_RATIO_LB']
        self.IT_FAN_AIRFLOW_RATIO_UB = servers['IT_FAN_AIRFLOW_RATIO_UB']
        self.IT_FAN_FULL_LOAD_V = servers['IT_FAN_FULL_LOAD_V']
        self.ITFAN_REF_V_RATIO = servers['ITFAN_REF_V_RATIO']
        self.ITFAN_REF_P = servers['ITFAN_REF_P']
        self.INLET_TEMP_RANGE = servers['INLET_TEMP_RANGE']

        ##################################################################
        #################### HVAC CONFIGURATION ##########################
        ##################################################################

        # Air parameters
        self.C_AIR = hvac['C_AIR']  # J/kg.K
        self.RHO_AIR = hvac['RHO_AIR']  # kg/m3

        # CRAC Unit paramters
        self.CRAC_SUPPLY_AIR_FLOW_RATE_pu = hvac['CRAC_SUPPLY_AIR_FLOW_RATE_pu']
        self.CRAC_REFRENCE_AIR_FLOW_RATE_pu = hvac['CRAC_REFRENCE_AIR_FLOW_RATE_pu']
        self.CRAC_FAN_REF_P = hvac['CRAC_FAN_REF_P']

        # Chiller Stats
        self.CHILLER_COP = hvac['CHILLER_COP_BASE']
        self.CW_PRESSURE_DROP = hvac['CW_PRESSURE_DROP'] #Pa 
        self.CW_WATER_FLOW_RATE = hvac['CW_WATER_FLOW_RATE'] #m3/s
        self.CW_PUMP_EFFICIENCY = hvac['CW_PUMP_EFFICIENCY'] #%
        self.CHILLER_COP_K = hvac['CHILLER_COP_K']
        self.CHILLER_COP_T_NOMINAL = hvac['CHILLER_COP_T_NOMINAL']

        # Cooling Tower parameters
        self.CT_FAN_REF_P = hvac['CT_FAN_REF_P']
        self.CT_REFRENCE_AIR_FLOW_RATE = hvac['CT_REFRENCE_AIR_FLOW_RATE']
        self.CT_PRESSURE_DROP = hvac['CT_PRESSURE_DROP'] #Pa 
        self.CT_WATER_FLOW_RATE = hvac['CT_WATER_FLOW_RATE']#m3/s
        self.CT_PUMP_EFFICIENCY = hvac['CT_PUMP_EFFICIENCY'] #%
    
    def _avg_pair_list(self, pair_list):
        avg_first = sum(p[0] for p in pair_list) / len(pair_list)
        avg_second = sum(p[1] for p in pair_list) / len(pair_list)
        return [avg_first, avg_second]


#References:
#[1]: Postema, Björn Frits. "Energy-efficient data centres: model-based analysis of power-performance trade-offs." (2018).
#[2]: Raghunathan, S., & Vk, M. (2014). Power management using dynamic power state transitions and dynamic voltage frequency scaling controls in virtualized server clusters. Turkish Journal of Electrical Engineering and Computer Sciences, 24(4). doi: 10.3906/elk-1403-264
#[3]: Sun, Kaiyu, et al. "Prototype energy models for data centers." Energy and Buildings 231 (2021): 110603.
#[4]: Breen, Thomas J., et al. "From chip to cooling tower data center modeling: Part I influence of server inlet temperature and temperature rise across cabinet." 2010 12th IEEE Intersociety Conference on Thermal and Thermomechanical Phenomena in Electronic Systems. IEEE, 2010.
#[5]: https://h2ocooling.com/blog/look-cooling-tower-fan-efficiences/#:~:text=The%20tower%20has%20been%20designed,of%200.42%20inches%20of%20water.
