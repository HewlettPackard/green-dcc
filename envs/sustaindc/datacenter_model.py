import os
import numpy as np
import math

class Server():
    def __init__(self, full_load_pwr=None, idle_pwr=None, gpu_full_load_pwr=None, gpu_idle_pwr=None, server_config=None): 
        """Server class in charge of the energy consumption and thermal calculations of the individual servers
            in a Rack.

        Args:
            full_load_pwr (float, optional): Power at full capacity for CPU.
            idle_pwr (float, optional): Power while idle for CPU.
            gpu_full_load_pwr (float, optional): Power at full capacity for GPU.
            gpu_idle_pwr (float, optional): Power while idle for GPU.
            server_config (config): Configuration for the DC.
        """
        self.server_config = server_config
        
        # CPU power parameters
        self.full_load_pwr = full_load_pwr if full_load_pwr is not None else self.server_config.HP_PROLIANT[0]
        self.idle_pwr = idle_pwr if idle_pwr is not None else self.server_config.HP_PROLIANT[1]
        
        # GPU power parameters
        self.gpu_full_load_pwr = gpu_full_load_pwr if gpu_full_load_pwr is not None else self.server_config.NVIDIA_V100[1]
        self.gpu_idle_pwr = gpu_idle_pwr if gpu_idle_pwr is not None else self.server_config.NVIDIA_V100[0]
        
        self.m_cpu = None
        self.c_cpu = None
        self.m_itfan = None
        self.c_itfan = None
        self.cpu_curve1()
        self.itfan_curve2()
        
        self.v_fan = None  # needed later for calculating outlet temperature
        self.itfan_v_ratio_at_inlet_temp = None
        self.total_DC_full_load = None

    def cpu_curve1(self,):
        """
        initialize the cpu power ratio curve at different IT workload ratios as a function of inlet temperatures [3]
        """
        # curve parameters at lowest ITE utilization 0%
        self.m_cpu = (self.server_config.CPU_POWER_RATIO_UB[0]-self.server_config.CPU_POWER_RATIO_LB[0])/(self.server_config.INLET_TEMP_RANGE[1]-self.server_config.INLET_TEMP_RANGE[0])
        self.c_cpu = self.server_config.CPU_POWER_RATIO_UB[0] - self.m_cpu*self.server_config.INLET_TEMP_RANGE[1]
        # max vertical shift in power ratio curve at a given point in inlet temperature for 100% change in ITE input load pct
        self.ratio_shift_max_cpu = self.server_config.CPU_POWER_RATIO_LB[1] - self.server_config.CPU_POWER_RATIO_LB[0]

    def itfan_curve2(self,):
        """
        initialize the itfan velocity ratio curve at different IT workload ratios as a function of inlet temperatures [3]
        """
        # curve parameters at ITE utilization 25%
        self.m_itfan = (self.server_config.IT_FAN_AIRFLOW_RATIO_UB[0]-self.server_config.IT_FAN_AIRFLOW_RATIO_LB[0])/(self.server_config.INLET_TEMP_RANGE[1]-self.server_config.INLET_TEMP_RANGE[0])
        self.c_itfan = self.server_config.IT_FAN_AIRFLOW_RATIO_UB[0] - self.m_itfan*self.server_config.INLET_TEMP_RANGE[1]
        # max vertical shift in fan flow ratio curve at a given point for 75% change in ITE input load pct
        self.ratio_shift_max_itfan = self.server_config.IT_FAN_AIRFLOW_RATIO_LB[1] - self.server_config.IT_FAN_AIRFLOW_RATIO_LB[0]
    
    def compute_instantaneous_cpu_pwr(self, inlet_temp, ITE_load_pct):
        """Calculate the CPU power consumption
        
        Args:
            inlet_temp (float): Inlet temperature
            ITE_load_pct (float): CPU utilization
            
        Returns:
            float: CPU power consumption in Watts
        """
        # Existing CPU power calculation method (assuming this was in the original code)
        base_cpu_power_ratio = self.m_cpu * inlet_temp + self.c_cpu
        cpu_power_ratio_at_inlet_temp = base_cpu_power_ratio + self.ratio_shift_max_cpu * (ITE_load_pct/100)
        cpu_power = max(self.idle_pwr, self.full_load_pwr * cpu_power_ratio_at_inlet_temp)
        return cpu_power

    def compute_instantaneous_fan_pwr(self, inlet_temp, ITE_load_pct):
        """Calculate the IT fan power consumption
        
        Args:
            inlet_temp (float): Inlet temperature
            ITE_load_pct (float): IT workload percentage
            
        Returns:
            float: Fan power consumption in Watts
        """
        # Existing fan power calculation method (assuming this was in the original code)
        base_itfan_v_ratio = self.m_itfan * inlet_temp + self.c_itfan
        itfan_v_ratio_at_inlet_temp = base_itfan_v_ratio + self.ratio_shift_max_itfan * (ITE_load_pct/100)
        self.itfan_v_ratio_at_inlet_temp = itfan_v_ratio_at_inlet_temp
        self.v_fan = self.server_config.IT_FAN_FULL_LOAD_V * itfan_v_ratio_at_inlet_temp
        itfan_pwr = self.server_config.ITFAN_REF_P * (itfan_v_ratio_at_inlet_temp/self.server_config.ITFAN_REF_V_RATIO)
        return itfan_pwr

    def compute_instantaneous_gpu_pwr(self, gpu_utilization):
        """Calculate GPU power based on utilization using the logarithmic model
        
        Args:
            gpu_utilization (float): GPU utilization percentage (0-100)
            
        Returns:
            float: GPU power consumption in Watts
        """

        # Apply the formula: y = α + β × log₂(1 + x)
        # where α is idle_power and β is (max_power - idle_power)
        # as referenced from from [6]
        alpha = self.gpu_idle_pwr
        beta = self.gpu_full_load_pwr - self.gpu_idle_pwr
        
        gpu_power = alpha + beta * math.log2(1 + gpu_utilization)
        return gpu_power


class Rack():
    def __init__(self, server_config_list, gpu_config_list=None, max_W_per_rack=10000, rack_config=None):
        """Defines the rack as a collection of servers

        Args:
            server_config_list (list): Server configuration
            gpu_config_list (list, optional): GPU configuration 
            max_W_per_rack (int): Maximum power allowed for a whole rack. Defaults to 10000.
            rack_config (config): Rack configuration. Defaults to None.
        """
        
        self.rack_config = rack_config
        
        self.server_list = []
        self.has_gpus = gpu_config_list is not None
        self.current_rack_load = 0
        
        for i, server_config in enumerate(server_config_list):
            # Get GPU info if available
            gpu_full_load_pwr = None
            gpu_idle_pwr = None
            
            if self.has_gpus and i < len(gpu_config_list):
                gpu_full_load_pwr = gpu_config_list[i]['full_load_pwr']
                gpu_idle_pwr = gpu_config_list[i]['idle_pwr']
            
            # Create server object with GPU info
            self.server_list.append(Server(
                full_load_pwr=server_config['full_load_pwr'], 
                idle_pwr=server_config['idle_pwr'],
                gpu_full_load_pwr=gpu_full_load_pwr,
                gpu_idle_pwr=gpu_idle_pwr,
                server_config=self.rack_config
            ))
            
            # Track power for capacity planning
            self.current_rack_load += self.server_list[-1].full_load_pwr
            if self.has_gpus and i < len(gpu_config_list):
                self.current_rack_load += self.server_list[-1].gpu_full_load_pwr
            
            if self.current_rack_load >= max_W_per_rack:
                self.server_list.pop()
                break
            
        self.num_servers = len(self.server_list)
        self.num_gpus = self.num_servers if self.has_gpus else 0
        self.server_and_fan_init()
        self.v_fan_rack = None
  
    def server_and_fan_init(self,):
        """
        Initialize the Server and Fan parameters for the servers in each rack with the specified data center configurations
        """
        
        #common to both cpu and fan 
        inlet_temp_lb, inlet_temp_ub = [], []
        
        # only for cpu
        m_cpu, c_cpu, ratio_shift_max_cpu, idle_pwr, full_load_pwr = [], [], [], [], []
            
        # only for it fan
        m_itfan, c_itfan, ratio_shift_max_itfan, ITFAN_REF_P, ITFAN_REF_V_RATIO, IT_FAN_FULL_LOAD_V = [], [], [], [], [], []
        
        # GPU parameters
        gpu_idle_pwr, gpu_full_load_pwr = [], []
        
        self.m_coefficient = 10 #1 -> 10 
        self.c_coefficient = 5 #1 -> 5
        self.it_slope = 20 #100 -> 20
            
        for server_item in self.server_list:
            
            #common to both cpu and fan
            inlet_temp_lb.append(server_item.server_config.INLET_TEMP_RANGE[0])
            inlet_temp_ub.append(server_item.server_config.INLET_TEMP_RANGE[1])
            
            # only for cpu
            m_cpu.append(server_item.m_cpu)
            c_cpu.append(server_item.c_cpu)
            ratio_shift_max_cpu.append(server_item.ratio_shift_max_cpu)
            idle_pwr.append(server_item.idle_pwr)
            full_load_pwr.append(server_item.full_load_pwr)
            
            # for GPU
            if self.has_gpus:
                gpu_idle_pwr.append(server_item.gpu_idle_pwr)
                gpu_full_load_pwr.append(server_item.gpu_full_load_pwr)
            
            # only for itfan
            m_itfan.append(server_item.m_itfan)
            c_itfan.append(server_item.c_itfan)
            ratio_shift_max_itfan.append(server_item.ratio_shift_max_itfan)
            ITFAN_REF_P.append(server_item.server_config.ITFAN_REF_P)
            ITFAN_REF_V_RATIO.append(server_item.server_config.ITFAN_REF_V_RATIO)
            IT_FAN_FULL_LOAD_V.append(server_item.server_config.IT_FAN_FULL_LOAD_V)
            
        # common to both cpu and itfan
        self.inlet_temp_lb, self.inlet_temp_ub = np.array(inlet_temp_lb), np.array(inlet_temp_ub)
        
        # only for cpu
        self.m_cpu, self.c_cpu, self.ratio_shift_max_cpu, self.idle_pwr, self.full_load_pwr = \
            np.array(m_cpu), np.array(c_cpu), np.array(ratio_shift_max_cpu), np.array(idle_pwr), np.array(full_load_pwr)
        
        # for GPU
        if self.has_gpus:
            self.gpu_idle_pwr, self.gpu_full_load_pwr = np.array(gpu_idle_pwr), np.array(gpu_full_load_pwr)
        
        # only for itfan
        self.m_itfan, self.c_itfan, self.ratio_shift_max_itfan, self.ITFAN_REF_P, self.ITFAN_REF_V_RATIO, self.IT_FAN_FULL_LOAD_V = \
            np.array(m_itfan), np.array(c_itfan), np.array(ratio_shift_max_itfan), \
            np.array(ITFAN_REF_P), np.array(ITFAN_REF_V_RATIO), np.array(IT_FAN_FULL_LOAD_V) 
            
    def compute_instantaneous_pwr(self, inlet_temp, ITE_load_pct, GPU_load_pct=0):
        """Calculate the power consumption of the whole rack at the current step

        Args:
            inlet_temp (float): Room temperature
            ITE_load_pct (float): Current CPU usage
            GPU_load_pct (float): Current GPU usage (optional, defaults to 0)

        Returns:
            tuple: (cpu_power, itfan_power, gpu_power)
        """
        # Server CPU power
        server = self.server_list[0]
        tot_cpu_pwr = server.compute_instantaneous_cpu_pwr(inlet_temp, ITE_load_pct) * self.num_servers

        # GPU power calculation
        tot_gpu_pwr = 0
        if self.has_gpus and GPU_load_pct > 0:
            for server_item in self.server_list:
                tot_gpu_pwr += server_item.compute_instantaneous_gpu_pwr(GPU_load_pct)

        # IT fan power calculation (consider both CPU and GPU heat)
        tot_itfan_pwr = []
        for server_item in self.server_list:
            # Fan responds to highest thermal load between CPU and GPU
            effective_load = max(ITE_load_pct, GPU_load_pct) if self.has_gpus else ITE_load_pct
            tot_itfan_pwr.append(server_item.compute_instantaneous_fan_pwr(inlet_temp, effective_load))

        return tot_cpu_pwr, np.array(tot_itfan_pwr).sum(), tot_gpu_pwr

    def compute_instantaneous_pwr_vecd(self, inlet_temp, ITE_load_pct, GPU_load_pct=0):
        """Calculate the power consumption of the whole rack at the current step in a vectorized manner

        Args:
            inlet_temp (float): Room temperature
            ITE_load_pct (float): Current CPU usage
            GPU_load_pct (float): Current GPU usage (optional, defaults to 0)

        Returns:
            tuple: (cpu_power, itfan_power, gpu_power)
        """
        # CPU power calculation
        base_cpu_power_ratio = (self.m_cpu+0.05)*inlet_temp + self.c_cpu
        cpu_power_ratio_at_inlet_temp = base_cpu_power_ratio + self.ratio_shift_max_cpu*(ITE_load_pct/100)
        temp_arr = np.concatenate((self.idle_pwr.reshape(1,-1),
                               (self.full_load_pwr*cpu_power_ratio_at_inlet_temp).reshape(1,-1)),
                              axis=0)
        cpu_power = np.max(temp_arr, axis=0)

        # Memory power calculation and add to CPU power
        
        # GPU power calculation
        gpu_power = np.zeros_like(cpu_power) if self.has_gpus else np.zeros(1)
        if self.has_gpus and GPU_load_pct > 0:
            # Normalize utilization for log formula
            x = GPU_load_pct / 100.0
            # Vectorized calculation of the logarithmic formula
            alpha = self.gpu_idle_pwr
            beta = self.gpu_full_load_pwr - self.gpu_idle_pwr
            gpu_power = alpha + beta * np.log2(1 + x)
        
        # IT fan power calculation - respond to the highest heat load
        # Add heats generated
        effective_load = ITE_load_pct + GPU_load_pct if self.has_gpus else ITE_load_pct
        base_itfan_v_ratio = self.m_itfan*self.m_coefficient*inlet_temp + self.c_itfan*self.c_coefficient
        itfan_v_ratio_at_inlet_temp = base_itfan_v_ratio + self.ratio_shift_max_itfan*(effective_load/self.it_slope)
        itfan_pwr = self.ITFAN_REF_P * (itfan_v_ratio_at_inlet_temp/self.ITFAN_REF_V_RATIO)
        self.v_fan_rack = self.IT_FAN_FULL_LOAD_V*itfan_v_ratio_at_inlet_temp
        
        return np.sum(cpu_power), np.sum(itfan_pwr), np.sum(gpu_power)

    def get_average_rack_fan_v(self,):
        """Calculate the average fan velocity for each rack

        Returns:
            (float): Average fan flow rate for the rack
        """
            
        return self.v_fan_rack[0]
    
    def get_total_rack_fan_v(self,):
        """Calculate the total fan velocity for each rack

        Returns:
            (float): Total fan flow rate for the rack
        """
        return np.sum(self.v_fan_rack)
    
    def get_current_rack_load(self,):
        """Returns the total power consumption of the rack

        Returns:
            float: Total power consumption of the rack
        """
        return self.current_rack_load

    def clamp_supply_approach_temp(self,supply_approach_temperature):
        """Returns the clamped delta/ supply approach temperature between the range [3.8, 5.3]

        Returns:
            float: Supply approach temperature
        """
        return max(3.8, min(supply_approach_temperature, 5.3))


class DataCenter_ITModel():
    def __init__(self, num_racks, dc_memory, rack_supply_approach_temp_list, rack_CPU_config, rack_GPU_config=None, max_W_per_rack=10000, DC_ITModel_config=None, chiller_sizing=False):
        """Creates the DC from a giving DC configuration

        Args:
            num_racks (int): Number of racks in the DC
            rack_supply_approach_temp_list (list[float]): models the supply approach temperature for each rack based on geometry and estimated from CFD
            rack_CPU_config (list[list[dict]]): A list of lists where each list is associated with a rack. 
                It is a list of dictionaries with their full load and idle load values in W
            rack_GPU_config (list[list[dict]], optional): Similar structure as rack_CPU_config but for GPUs
            max_W_per_rack (int): Maximum power allowed for a whole rack. Defaults to 10000.
            DC_ITModel_config (config): Data center configuration. Defaults to None.
            chiller_sizing (bool): Whether to perform Chiller Power Sizing. Defaults to False.
        """
        self.DC_ITModel_config = DC_ITModel_config
        self.racks_list = []
        self.rack_supply_approach_temp_list = rack_supply_approach_temp_list
        self.rack_CPU_config = rack_CPU_config
        self.rack_GPU_config = rack_GPU_config
        self.dc_memory = dc_memory
        self.has_gpus = rack_GPU_config is not None
        
        self.rackwise_inlet_temp = []
        
        for i in range(num_racks):
            if self.has_gpus and i < len(self.rack_GPU_config):
                self.racks_list.append(Rack(
                    self.rack_CPU_config[i], 
                    self.rack_GPU_config[i], 
                    max_W_per_rack=max_W_per_rack, 
                    rack_config=self.DC_ITModel_config
                ))
            else:
                self.racks_list.append(Rack(
                    self.rack_CPU_config[i], 
                    max_W_per_rack=max_W_per_rack, 
                    rack_config=self.DC_ITModel_config
                ))
        
        self.total_datacenter_full_load()
        
        self.tower_flow_rate = np.round(DC_ITModel_config.CT_WATER_FLOW_RATE * 3600, 4)  # m³/hr
        self.hot_water_temp = None  # °C
        self.cold_water_temp = None  # °C
        self.wet_bulb_temp = None  # °C
        self.cycles_of_concentration = 5
        self.drift_rate = 0.01
        
    def compute_datacenter_IT_load_outlet_temp(self, ITE_load_pct_list, CRAC_setpoint, GPU_load_pct_list=None):
        """Calculate the average outlet temperature of all the racks

        Args:
            ITE_load_pct_list (List[float]): CPU load for each rack
            CRAC_setpoint (float): CRAC setpoint
            GPU_load_pct_list (List[float], optional): GPU load for each rack. Defaults to None.

        Returns:
            tuple: (rackwise_cpu_pwr, rackwise_itfan_pwr, rackwise_gpu_pwr, rackwise_outlet_temp)
        """
        # Default GPU load to 0 if not provided and GPUs exist
        if GPU_load_pct_list is None:
            GPU_load_pct_list = [0] * len(ITE_load_pct_list) if self.has_gpus else None

        rackwise_cpu_pwr = [] 
        rackwise_itfan_pwr = []
        rackwise_gpu_pwr = []
        rackwise_outlet_temp = []
        rackwise_inlet_temp = []
                
        c = 1.918
        d = 1.096
        e = 0.824
        f = 0.526
        g = -14.01
        
        for i, (rack, rack_supply_approach_temp, ITE_load_pct) in enumerate(
                zip(self.racks_list, self.rack_supply_approach_temp_list, ITE_load_pct_list)):
            
            #clamp supply approach temperatures
            rack_supply_approach_temp = rack.clamp_supply_approach_temp(rack_supply_approach_temp)
            rack_inlet_temp = rack_supply_approach_temp + CRAC_setpoint 
            rackwise_inlet_temp.append(rack_inlet_temp)
            
            # Get GPU load if applicable
            gpu_load = GPU_load_pct_list[i] if GPU_load_pct_list is not None else 0
            
            # Calculate power with GPU if applicable
            # Add cpu power and memory power
            # Memory energy consumption with respect to memory usage
            rack_cpu_power, rack_itfan_power, rack_gpu_power = rack.compute_instantaneous_pwr_vecd(
                rack_inlet_temp, ITE_load_pct, gpu_load)
            
            rackwise_cpu_pwr.append(rack_cpu_power)
            rackwise_itfan_pwr.append(rack_itfan_power)
            rackwise_gpu_pwr.append(rack_gpu_power)
            
            # Use total power (CPU + GPU + fan) for thermal calculations
            # The coefficient for scaling background power consumption of DRAM memory is referenced by approximation from from Figure 2 in [7]
            memory_power = 0.07*self.dc_memory
            total_power = rack_cpu_power + rack_itfan_power + rack_gpu_power
            
            power_term = total_power**d
            airflow_term = self.DC_ITModel_config.C_AIR*self.DC_ITModel_config.RHO_AIR*rack.get_total_rack_fan_v()**e * f
            
            outlet_temp = rack_inlet_temp + c * power_term / airflow_term + g 
            if outlet_temp > 60:
                print(f'WARNING, the outlet temperature is higher than 60C: {outlet_temp:.3f}')
            
            if outlet_temp - rack_inlet_temp < 2:
                print(f'There is something wrong with the delta calculation because is too small: {outlet_temp - rack_inlet_temp:.3f}')
                print(f'Inlet Temp: {rack_inlet_temp:.3f}, Util: {ITE_load_pct}, GPU Util: {gpu_load}, Total Power: {total_power:.3f}')
                print(f'Power term: {power_term:.3f}, Airflow term: {airflow_term:.3f}')
                print(f'Delta: {c * power_term / airflow_term + g:.3f}')
                raise Exception("Sorry, no numbers below 2")

            if (ITE_load_pct > 95 or (gpu_load > 95 and self.has_gpus)) and CRAC_setpoint < 16.5:
                if outlet_temp - rack_inlet_temp < 2:
                    print(f'There is something wrong with the delta calculation for MAX is too small: {outlet_temp - rack_inlet_temp:.3f}')
                    print(f'Inlet Temp: {rack_inlet_temp:.3f}, Total Power: {total_power:.3f}')
                    print(f'Power term: {power_term:.3f}, Airflow term: {airflow_term:.3f}')
                    print(f'Delta: {c * power_term / airflow_term + g:.3f}')
                    raise Exception("Sorry, no numbers below 2")
                    
            rackwise_outlet_temp.append(outlet_temp)

        self.rackwise_inlet_temp = rackwise_inlet_temp
            
        return rackwise_cpu_pwr, rackwise_itfan_pwr, memory_power, rackwise_gpu_pwr, rackwise_outlet_temp
    
    def total_datacenter_full_load(self,):
        """Calculate the total DC IT power consumption (CPU, GPU, and fan)
        """
        total_power = sum(rack.get_current_rack_load() for rack in self.racks_list)
        self.total_DC_full_load = total_power

    def calculate_cooling_tower_water_usage(self):
        """
        Calculate the estimated water usage of the cooling tower.

        This function uses the attributes set in the class to estimate the water usage based 
        [Sharma, R.K., Shah, A., Bash, C.E., Christian, T., & Patel, C.D. (2009). Water efficiency management in datacenters: Metrics and methodology. 2009 IEEE International Symposium on Sustainable Systems and Technology, 1-6.]
        [Mohammed Shublaq, Ahmad K. Sleiti., (2020).  Experimental analysis of water evaporation losses in cooling towers using filters]
        https://spxcooling.com/water-calculator/
        """
        # We're assuming m³/hr, which is standard

        # Calculate the range (difference between hot and cold water temperature)
        range_temp = self.hot_water_temp - self.cold_water_temp
        
        y_intercept = 0.3528 * range_temp + 0.101
        
        # The water usage estimation formula would need to be derived from the graph you provided.
        
        norm_water_usage = 0.044 * self.wet_bulb_temp + y_intercept
        
        water_usage = np.clip(norm_water_usage, 0, None)
        
        water_usage += water_usage * self.drift_rate  # adjust for drift

        # Convert m³/hr to the desired unit (e.g., liters per 15 minutes) if necessary
        # There are 1000 liters in a cubic meter. There are 4 15-minute intervals in an hour.
        water_usage_liters_per_15min = np.round((water_usage * 1000) / 4, 4)

        return water_usage_liters_per_15min

def calculate_chiller_power(max_cooling_cap, load, ambient_temp):
    """
    Calculate the chiller power consumption based on load and operating conditions.
    
    Obtained from:
        1) https://github.com/NREL/EnergyPlus/blob/9bb39b77a871dee7543c892ae53b0812c4c17b0d/testfiles/AirCooledElectricChiller.idf
        2) https://github.com/NREL/EnergyPlus/issues/763
        3) https://dmey.github.io/EnergyPlusFortran-Reference/proc/calcelectricchillermodel.html
        4) https://github.com/NREL/EnergyPlus/blob/9bb39b77a871dee7543c892ae53b0812c4c17b0d/tst/EnergyPlus/unit/ChillerElectric.unit.cc#L95

    Args:
        max_cooling_cap (float): Maximum cooling capacity of the chiller (Watts).
        load (float): The heat load to be removed by the chiller (Watts).
        ambient_temp (float): Current ambient temperature (Celsius).
        
    Returns:
        float: Estimated power consumption of the chiller (Watts).
    """
    
    # Coefficients from https://github.com/NREL/EnergyPlus/blob/9bb39b77a871dee7543c892ae53b0812c4c17b0d/tst/EnergyPlus/unit/ChillerElectric.unit.cc#L95
    capacity_coefficients = [0.94483600, -0.05700880, 0.00185486]
    power_coefficients = [2.333, -1.975, 0.6121]
    full_load_factor = [0.03303, 0.6852, 0.2818]

    # Chiller design specifications
    min_plr = 0.05
    max_plr = 1.0
    design_cond_temp = 35.0
    design_evp_out_temp = 6.67
    chiller_nominal_cap = max_cooling_cap
    temp_rise_coef = 2.778
    rated_cop = 3.0
    
    # Calculate the delta temperature for capacity adjustment
    delta_temp = (ambient_temp - design_cond_temp)/temp_rise_coef - (design_evp_out_temp - design_cond_temp)
    
    # Calculate available nominal capacity ratio
    avail_nom_cap_rat = capacity_coefficients[0] + capacity_coefficients[1] * delta_temp + capacity_coefficients[2] * delta_temp**2
    
    # Calculate available chiller capacity
    available_capacity = chiller_nominal_cap * avail_nom_cap_rat if avail_nom_cap_rat != 0 else 0

    # Calculate power ratio
    full_power_ratio = power_coefficients[0] + power_coefficients[1] * avail_nom_cap_rat + power_coefficients[2] * avail_nom_cap_rat**2
    
    # Determine part load ratio (PLR)
    part_load_ratio = max(min_plr, min(load / available_capacity, max_plr)) if available_capacity > 0 else 0
    
    # Calculate fractional full load power
    frac_full_load_power = full_load_factor[0] + full_load_factor[1] * part_load_ratio + full_load_factor[2] * part_load_ratio**2
    
    # Determine operational part load ratio (OperPartLoadRat)
    # If the PLR is less than Min PLR calculate the actual PLR for calculations. The power will then adjust for the cycling.
    if available_capacity > 0:
        if load / available_capacity < min_plr:
            oper_part_load_rat = load / available_capacity
        else:
            oper_part_load_rat = part_load_ratio
    else:
        oper_part_load_rat = 0.0
    
    # Operational PLR for actual conditions
    if oper_part_load_rat < min_plr:
        frac = min(1.0, oper_part_load_rat / min_plr)
    else:
        frac = 1.0

    # Calculate the chiller compressor power
    power = frac_full_load_power * full_power_ratio * available_capacity / rated_cop * frac

    #  Total heat rejection is the sum of the cooling capacity and the power input
    total_heat_rejection = load + power

    return power if oper_part_load_rat > 0 else 0


def calculate_HVAC_power(CRAC_setpoint, avg_CRAC_return_temp, ambient_temp, data_center_full_load, DC_Config, ctafr=None):
    """Calculate the HVAC power attributes

        Args:
            CRAC_Setpoint (float): The control action
            avg_CRAC_return_temp (float): The average of the temperatures from all the Racks + their corresponding return approach temperature (Delta)
            ambient_temp (float): outside air temperature
            data_center_full_load (float): total data center capacity

        Returns:
            CRAC_Fan_load (float): CRAC fan power
            CT_Fan_pwr (float):  Cooling tower fan power
            CRAC_cooling_load (float): CRAC cooling load
            Compressor_load (float): Chiller compressor load
        """
    # Air system calculations
    m_sys = DC_Config.RHO_AIR * DC_Config.CRAC_SUPPLY_AIR_FLOW_RATE_pu * data_center_full_load
    CRAC_cooling_load = m_sys * DC_Config.C_AIR * max(0.0, avg_CRAC_return_temp - CRAC_setpoint) # coo.Q_thistime
    CRAC_Fan_load = DC_Config.CRAC_FAN_REF_P * (DC_Config.CRAC_SUPPLY_AIR_FLOW_RATE_pu / DC_Config.CRAC_REFRENCE_AIR_FLOW_RATE_pu)**3
    
    chiller_power = calculate_chiller_power(DC_Config.CT_FAN_REF_P, CRAC_cooling_load, ambient_temp)

    # Chiller power calculation
    power_consumed_CW = (DC_Config.CW_PRESSURE_DROP * DC_Config.CW_WATER_FLOW_RATE) / DC_Config.CW_PUMP_EFFICIENCY

    # Chilled water pump power calculation
    power_consumed_CT = (DC_Config.CT_PRESSURE_DROP*DC_Config.CT_WATER_FLOW_RATE)/DC_Config.CT_PUMP_EFFICIENCY

    if ambient_temp < 5:
        return CRAC_Fan_load, 0.0, CRAC_cooling_load, chiller_power, power_consumed_CW, power_consumed_CT

    # Cooling tower fan power calculations
    Cooling_tower_air_delta = max(50 - (ambient_temp - CRAC_setpoint), 1)
    m_air = CRAC_cooling_load / (DC_Config.C_AIR * Cooling_tower_air_delta)
    v_air = m_air / DC_Config.RHO_AIR
    
    # Reference cooling tower air flow rate
    if ctafr is None:
        ctafr = DC_Config.CT_REFRENCE_AIR_FLOW_RATE
    CT_Fan_pwr = DC_Config.CT_FAN_REF_P * (min(v_air / ctafr, 1))**3
    
    # ToDo: exploring the new chiller_power method
    return CRAC_Fan_load, CT_Fan_pwr, CRAC_cooling_load, chiller_power, power_consumed_CW, power_consumed_CT

def chiller_sizing(DC_Config, dc_memory, min_CRAC_setpoint=16, max_CRAC_setpoint=22, max_ambient_temp=40.0):
    '''
    Calculates the chiller sizing for a data center based on the given configuration and parameters.
    
    Parameters:
        DC_Config (object): The data center configuration object.
        dc_memory (float): the total available memory in the datacenter.
        min_CRAC_setpoint (float): The minimum CRAC setpoint temperature in degrees Celsius. Default is 16.
        max_CRAC_setpoint (float): The maximum CRAC setpoint temperature in degrees Celsius. Default is 22.
        max_ambient_temp (float): The maximum ambient temperature in degrees Celsius. Default is 40.0.
    
    Returns:
        tuple: A tuple containing the cooling tower reference air flow rate (ctafr) and the rated load of the cooling tower (CT_rated_load).
    '''
    # Create GPU configs if available in the DC_Config
    gpu_config = None
    if hasattr(DC_Config, 'RACK_GPU_CONFIG'):
        gpu_config = DC_Config.RACK_GPU_CONFIG
    
    dc = DataCenter_ITModel(num_racks=DC_Config.NUM_RACKS,
                            dc_memory=dc_memory,
                            rack_supply_approach_temp_list=DC_Config.RACK_SUPPLY_APPROACH_TEMP_LIST,
                            rack_CPU_config=DC_Config.RACK_CPU_CONFIG,
                            rack_GPU_config=gpu_config,
                            max_W_per_rack=DC_Config.MAX_W_PER_RACK,
                            DC_ITModel_config=DC_Config)
    
    # Set maximum load for both CPU and GPU if present
    cpu_load = 100.0
    ITE_load_pct_list = [cpu_load for i in range(DC_Config.NUM_RACKS)]
    
    # Set GPU load if GPU config is available
    GPU_load_pct_list = None
    if gpu_config:
        gpu_load = 100.0
        GPU_load_pct_list = [gpu_load for i in range(DC_Config.NUM_RACKS)]
    
    # Calculate with both CPU and GPU loads if GPU is present
    result = dc.compute_datacenter_IT_load_outlet_temp(
        ITE_load_pct_list=ITE_load_pct_list, 
        CRAC_setpoint=max_CRAC_setpoint,
        GPU_load_pct_list=GPU_load_pct_list
    )
    
    # Unpack result
    if len(result) == 5:  # Includes GPU power
        rackwise_cpu_pwr, rackwise_itfan_pwr, memory_power, rackwise_gpu_pwr, rackwise_outlet_temp = result
    else:  # No GPU
        rackwise_cpu_pwr, rackwise_itfan_pwr, memory_power, rackwise_outlet_temp = result
        rackwise_gpu_pwr = [0] * len(rackwise_cpu_pwr)
    
    avg_CRAC_return_temp = calculate_avg_CRAC_return_temp(
        rack_return_approach_temp_list=DC_Config.RACK_RETURN_APPROACH_TEMP_LIST,
        rackwise_outlet_temp=rackwise_outlet_temp
    )
    
    # Calculate total power including GPU if present
    # 0.07 is the scaling factor for background power consumption in DRAM memory as discussed in [7]
    data_center_total_ITE_Load = sum(rackwise_cpu_pwr) + sum(rackwise_itfan_pwr) + sum(rackwise_gpu_pwr) + memory_power
    
    m_sys = DC_Config.RHO_AIR * DC_Config.CRAC_SUPPLY_AIR_FLOW_RATE_pu * data_center_total_ITE_Load
    
    CRAC_cooling_load = m_sys*DC_Config.C_AIR*max(0.0, avg_CRAC_return_temp-min_CRAC_setpoint) 
    Cooling_tower_air_delta = max(50 - (max_ambient_temp-min_CRAC_setpoint), 1)  
    
    m_air = CRAC_cooling_load/(DC_Config.C_AIR*Cooling_tower_air_delta) 
    
    # Cooling Tower Reference Air FlowRate
    ctafr = m_air/DC_Config.RHO_AIR
    
    CT_rated_load = CRAC_cooling_load
    
    return ctafr, CT_rated_load
    
def calculate_avg_CRAC_return_temp(rack_return_approach_temp_list, rackwise_outlet_temp):   
    """Calculate the CRAC return air temperature

        Args:
            rack_return_approach_temp_list (List[float]): The delta change in temperature from each rack to the CRAC unit
            rackwise_outlet_temp (float): The outlet temperature of each rack
        Returns:
            (float): CRAC return air temperature
        """
    n = len(rack_return_approach_temp_list)
    return sum([i + j for i,j in zip(rack_return_approach_temp_list, rackwise_outlet_temp)])/n  # CRAC return is averaged across racks

"""
References:
[1]: Postema, Björn Frits. "Energy-efficient data centres: model-based analysis of power-performance trade-offs." (2018).
[2]: Raghunathan, S., & Vk, M. (2014). Power management using dynamic power state transitions and dynamic voltage frequency
     scaling controls in virtualized server clusters. Turkish Journal of Electrical Engineering and Computer Sciences, 24(4). doi: 10.3906/elk-1403-264
[3]: Sun, Kaiyu, et al. "Prototype energy models for data centers." Energy and Buildings 231 (2021): 110603.
[4]: Breen, Thomas J., et al. "From chip to cooling tower data center modeling: Part I influence of server inlet temperature and temperature 
     rise across cabinet." 2010 12th IEEE Intersociety Conference on Thermal and Thermomechanical Phenomena in Electronic Systems. IEEE, 2010.
[5]: https://h2ocooling.com/blog/look-cooling-tower-fan-efficiences/#:~:text=The%20tower%20has%20been%20designed,of%200.42%20inches%20of%20water.
[6]: X. Tang and Z. Fu, "CPU–GPU Utilization Aware Energy-Efficient Scheduling Algorithm on Heterogeneous Computing Systems," in IEEE Access, vol. 8, pp. 58948-58958, 2020, doi: 10.1109/ACCESS.2020.2982956. 
[7]: Seunghak Lee, Ki-Dong Kang, Hwanjun Lee, Hyungwon Park, Younghoon Son, Nam Sung Kim, and Daehoon Kim. 2021. GreenDIMM: OS-assisted DRAM Power Management for DRAM with a Sub-array Granularity Power-Down State. In MICRO-54: 54th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO '21). Association for Computing Machinery, New York, NY, USA, 131–142. https://doi.org/10.1145/3466752.3480089.
"""



