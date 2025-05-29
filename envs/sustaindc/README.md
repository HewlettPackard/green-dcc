# SustainDC: Detailed Data Center Simulation Model

This document describes the internal thermal and electrical simulation model used for each data center within the SustainCluster benchmark. It details how IT load translates to power consumption and heat, and how the cooling system responds.

This model is integrated as a sub-component within the `TaskSchedulingEnv` and `DatacenterClusterManager`. It provides the physics-based calculations that underpin the energy, carbon, and cost metrics used for evaluating scheduling policies.

---

## 1. Overview

Each simulated SustainDC instance models the dynamic interplay between:
*   **IT Load:** Power consumption arising from CPU, GPU, and Memory usage based on scheduled tasks.
*   **IT Fan Power:** Power required by server fans to dissipate heat, responding to IT load and temperatures.
*   **Thermal Dynamics:** Heat generation by IT components, airflow within racks, and resulting server outlet temperatures.
*   **HVAC System:** Energy consumed by the cooling infrastructure (CRAC units, Chillers, Pumps, Cooling Towers) needed to remove the generated heat, influenced by ambient weather conditions.
*   **Carbon & Pricing:** Uses external managers (`CIManager`, `PriceManager`) to associate energy consumption with real-time carbon emissions and electricity costs.
*   **Water Usage:** Estimates cooling tower water consumption.

The model represents a data center comprising multiple `Racks`, each containing numerous `Servers` (which encapsulate CPU, GPU, Memory, and Fan components). An `HVAC system` provides cooling.

---

## 2. Integration within SustainCluster

*   The `DatacenterClusterManager` holds multiple `SustainDC` instances.
*   The `SustainDC` environment wrapper manages task queues and resource allocation (`total_cores`, `total_gpus`, `total_mem` specified in `datacenters.yaml`).
*   The core physics calculations occur within the `Datacenter_ITModel` class (`envs/sustaindc/datacenter_model.py`) and associated helper functions (`calculate_HVAC_power`, `calculate_chiller_power`).
*   At each simulation step, the manager provides the current workload (CPU%, GPU%, MEM%) to the model, which then calculates power, temperatures, and cooling requirements.

---

## 3. Resource Model

*   Total computational resources (`total_cores`, `total_gpus`, `total_mem`) are defined per-DC in `datacenters.yaml`.
*   Incoming tasks specify requirements for these resources.
*   The `SustainDC` environment ensures tasks are only scheduled if sufficient resources are available.
*   Utilization percentages are passed to the power models below.

---

## 4. IT Power and Thermal Modeling (`datacenter_model.py`)

A bottom-up approach models power consumption and heat generation from individual components.

### 4.1 CPU Power Model (`Server.compute_instantaneous_cpu_pwr`)

CPU power is modeled based on utilization and inlet temperature, adapting efficiency curves from \cite{Sun2021Prototype}.
```math
P_{CPU\_base\_ratio} = m_{cpu} \cdot T_{in} + c_{cpu}
```
```math
P_{CPU\_ratio} = P_{CPU\_base\_ratio} + \Delta_{ratio\_max\_cpu} \cdot \frac{\text{CPU Load \%}}{100}
```
```math
P_{CPU} = \max(P_{idle\_cpu}, P_{full\_cpu} \cdot P_{CPU\_ratio})
```
Where $P_{idle\_cpu}$, $P_{full\_cpu}$, slope ($m_{cpu}$), intercept ($c_{cpu}$), and max ratio shift ($\Delta_{ratio\_max\_cpu}$) are derived from server characteristics defined in `dc_config.json` (e.g., `HP_PROLIANT` defaults, `CPU_POWER_RATIO` bounds).

### 4.2 GPU Power Model (`Server.compute_instantaneous_gpu_pwr`)

GPU power consumption follows a logarithmic model based on utilization \cite{Tang2020CPU}:
```math
P_{GPU} = P_{idle\_gpu} + (P_{full\_gpu} - P_{idle\_gpu}) \cdot \log_2(1 + \frac{\text{GPU Load \%}}{100})
```
Where $P_{idle\_gpu}$ and $P_{full\_gpu}$ are defined per server/GPU type in `dc_config.json` (e.g., `NVIDIA_V100` defaults).

### 4.3 Memory Power Model (`DataCenter_ITModel.compute_datacenter_IT_load_outlet_temp`)

A background power consumption for DRAM is estimated based on the total data center memory capacity, distributed evenly across racks \cite{Lee2021GreenDIMM}:
```math
P_{Mem\_per\_rack} = 0.07 \frac{\text{W}}{\text{GB}} \times \frac{\text{Total DC Memory (GB)}}{\text{Number of Racks}}
```
*(Note: This is a simplified background load and does not currently scale dynamically with memory access patterns).*

### 4.4 IT Fan Power Model (`Server.compute_instantaneous_fan_pwr`, used within `Rack.compute_instantaneous_pwr_vecd`)

Server fan power scales cubically with the required fan speed (airflow ratio). Crucially, the required fan speed ratio (`itfan_v_ratio_at_inlet_temp`) depends on both the **inlet air temperature** (`T_in`) and an **effective load percentage**, which represents the overall thermal load from active components. In the current implementation (`Rack.compute_instantaneous_pwr_vecd`), this effective load is calculated as the *average* of the CPU, GPU (if present), and Memory utilization percentages:
```math
\text{Effective Load \%} = \text{average}(\text{CPU Load \%}, \text{GPU Load \%}, \text{Memory Load \%})
```
The fan speed ratio is then determined using temperature-dependent curves similar to the CPU model:
```math
V_{Fan\_base\_ratio} = m_{itfan} \cdot T_{in} + c_{itfan}
```
```math
V_{Fan\_ratio} = V_{Fan\_base\_ratio} + \Delta_{ratio\_max\_itfan} \cdot \frac{\text{Effective Load \%}}{\text{Slope Factor}}
```
```math
P_{IT\_Fan} = P_{ref\_fan} \cdot \left(\frac{V_{Fan\_ratio}}{V_{ref\_fan\_ratio}}\right)^3
```
Parameters ($m_{itfan}, c_{itfan}, \Delta_{ratio\_max\_itfan}$, reference powers/ratios, slope factor) are defined in `dc_config.json`.

### 4.5 Total IT Power (`Rack.compute_instantaneous_pwr_vecd`, `DataCenter_ITModel.compute_datacenter_IT_load_outlet_temp`)

The total instantaneous IT power for the data center is the sum of all components across all racks:
```math
P_{IT\_total} = \sum_{r \in Racks} (P_{CPU}^{(r)} + P_{GPU}^{(r)} + P_{Memory}^{(r)} + P_{IT\_Fan}^{(r)})
```

### 4.6 Rack Outlet Temperature (`DataCenter_ITModel.compute_datacenter_IT_load_outlet_temp`)

The temperature of the air exiting each rack (`T_{out}`) is calculated based on the inlet temperature (`T_{in}`), the total power dissipated within the rack (`P_{Rack\_total} = P_{CPU} + P_{GPU} + P_{Memory} + P_{IT\_Fan}`), and the total airflow provided by the server fans (`V_{fan\_rack}`):
```math
P_{term} = P_{Rack\_total} ^ d
```
```math
V_{term} = (C_{air} \cdot \rho_{air} \cdot V_{fan\_rack} ^ e \cdot f)
```
```math
T_{out} = T_{in} + c \cdot \frac{P_{term}}{V_{term}} + g
```
Where $c, d, e, f, g$ are empirical coefficients defined within the code, $C_{air}$ and $\rho_{air}$ are properties of air. The inlet temperature $T_{in}$ itself depends on the CRAC setpoint and rack-specific supply approach temperatures. *(Note: The current implementation assumes uniform utilization and calculates based on one reference rack, scaling the results).*

---

## 5. HVAC System Modeling (`datacenter_model.py`)

The cooling system model calculates the energy required to remove the heat generated by $P_{IT\_total}$.

### 5.1 Cooling Load (`calculate_HVAC_power`)

The primary cooling load is determined by the need to cool the return air back down to the target CRAC setpoint temperature. The return air temperature (`avg_CRAC_return_temp`) is the average of outlet temperatures from all racks, adjusted by rack-specific return approach temperatures ($\Delta T_{rack}^{(i)}$ from `dc_config.json`):
```math
T_{return} = \frac{1}{N_{racks}} \sum_{i=1}^{N_{racks}} \left(T_{out}^{(i)} + \Delta T_{rack}^{(i)}\right)
```
The CRAC cooling load is then:
```math
Q_{CRAC} = \dot{m}_{air} \cdot C_{air} \cdot \max(0.0, T_{return} - T_{setpoint})
```
Where $\dot{m}_{air}$ is the total mass flow rate of air supplied by the CRAC units.

### 5.2 CRAC Fan Power (`calculate_HVAC_power`)

Similar to IT fans, CRAC fan power scales cubically with the required airflow rate relative to a reference point:
```math
P_{CRACfan} = P_{ref\_CRACfan} \cdot \left(\frac{\dot{V}_{CRAC}}{\dot{V}_{ref\_CRAC}}\right)^3
```

### 5.3 Chiller Power (`calculate_chiller_power`)

The chiller power is calculated using a detailed model based on performance curves and part-load ratio (PLR) logic derived from **EnergyPlus** examples \cite{EnergyPlusEngRef}. The model (`calculate_chiller_power` function) takes the current cooling load (`Q_{CRAC}`), the ambient outdoor temperature (`ambient_temp`), and the chiller's maximum design capacity as inputs. It calculates:
*   The chiller's available cooling capacity under current conditions (derating based on ambient temperature relative to design specs).
*   The required Part Load Ratio (PLR = `load / available_capacity`).
*   The power consumption using polynomial curves that relate efficiency (COP or power ratio) to PLR and available capacity ratio.
This captures the non-linear efficiency of chillers under varying load and environmental conditions. See code comments for links to specific EnergyPlus source files used as reference.

### 5.4 Cooling Tower and Pump Power (`calculate_HVAC_power`)

*   **Cooling Tower Fan Power ($P_{CTfan}$):** Scales cubically with the required airflow, which is calculated based on the cooling load and the temperature difference between ambient air and the CRAC setpoint.
*   **Pump Power ($P_{CWpump}$, $P_{CTpump}$):** Calculated based on required water flow rates, system pressure drops, and pump efficiencies specified in `dc_config.json`.

### 5.5 Total HVAC Power

```math
P_{HVAC\_total} = P_{CRACfan} + P_{chiller} + P_{CTfan} + P_{CWpump} + P_{CTpump}
```

---

## 6. Total Datacenter Power

The total instantaneous power demand of the data center is the sum of IT and HVAC power:
```math
P_{total} = P_{IT\_total} + P_{HVAC\_total}
```
This value is used with the current electricity price and carbon intensity to calculate operational cost and emissions for the timestep.

---

## 7. Water Usage Modeling (`DataCenter_ITModel.calculate_cooling_tower_water_usage`)

Estimated water loss (primarily evaporation) in the cooling tower is calculated based on the temperature difference between the hot water entering and cold water leaving the tower (`range_temp`) and the ambient wet-bulb temperature (`wet_bulb_temp`, derived from weather data), using empirical formulas adapted from \cite{Sharma2009Water, Shublaq2020Experimental}. Adjustments are made for drift loss. The result is reported in Liters per 15 minutes.

---

## 8. Configurability

Key parameters are configurable:
*   High-level resources (`total_cores`, `total_gpus`, `total_mem`) in `datacenters.yaml`.
*   Detailed physical parameters (rack layout, approach temps, component efficiencies, power characteristics for CPU/GPU types, HVAC specs) in the JSON file specified by `dc_config_file` (e.g., `configs/dcs/dc_config.json`).

---

## 9. Model References

The simulation models used in SustainCluster build upon established research and data sources in data center energy modeling, thermal management, and hardware characterization:

*   [1] **CPU/Fan Power & Thermal Curves:** Sun, K., et al. (2021). *Prototype energy models for data centers*. Energy and Buildings, 231, 110603. \[[DOI: 10.1016/j.enbuild.2020.110603](https://doi.org/10.1016/j.enbuild.2020.110603)\]
*   [2] **Integrated DC Modeling:** Breen, T. J., et al. (2010). *From chip to cooling tower data center modeling*. 12th IEEE Intersociety Conference on Thermal and Thermomechanical Phenomena in Electronic Systems (ITHERM). \[[DOI: 10.1109/ITHERM.2010.5501421](https://doi.org/10.1109/ITHERM.2010.5501421)\]
*   [3] **Original Curve Basis:** Postema, B. F. (2018). *Energy-efficient data centres: model-based analysis of power-performance trade-offs*. PhD Thesis, Delft University of Technology. \[[Repository Link](https://ris.utwente.nl/ws/portalfiles/portal/78047555/Postema_thesis_final_2.pdf)\]
*   [4] **Chiller Model Reference:** EnergyPlus™ Documentation and Engineering Reference. *EnergyPlus v25.1.0 Engineering Reference*. U.S. Department of Energy. (See code comments in `calculate_chiller_power` for specific source file links, e.g., \[[EnergyPlus GitHub](https://github.com/NREL/EnergyPlus)\])
*   [5] **GPU Power Model:** Tang, X., & Fu, Z. (2020). *CPU–GPU Utilization Aware Energy-Efficient Scheduling Algorithm on Heterogeneous Computing Systems*. IEEE Access, 8, 58948-58958. \[[DOI: 10.1109/ACCESS.2020.2982956](https://doi.org/10.1109/ACCESS.2020.2982956)\]
*   [6] **Memory Background Power:** Lee, S., et al. (2021). *GreenDIMM: OS-assisted DRAM Power Management...*. 54th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO). \[[DOI: 10.1145/3466752.3480089](https://doi.org/10.1145/3466752.3480089)\]
*   [7] **Water Usage Model Basis:** Sharma, R. K., et al. (2009). *Water efficiency management in datacenters: Metrics and methodology*. IEEE International Symposium on Sustainable Systems and Technology (ISSST). \[[DOI: 10.1109/ISSST.2009.5156773](https://doi.org/10.1109/ISSST.2009.5156773)\]
*   [8] **Water Usage Model Basis:** Shublaq, M., & Sleiti, A. K. (2020). *Experimental analysis of water evaporation losses in cooling towers using filters*. International Journal of Thermofluids, 7-8, 100048. \[[DOI: 10.1016/j.applthermaleng.2020.115418](https://doi.org/10.1016/j.applthermaleng.2020.115418)\]
*   [9] **Cooling Tower Fans (Informal):** Industry data or studies on cooling tower fan efficiencies and power consumption. \[[White paper link](https://spxcooling.com/wp-content/uploads/CTII-01A.pdf)\]

---

## 10. Current Implementation Notes & Limitations

*   **Uniform Utilization Assumption:** The primary method `compute_datacenter_IT_load_outlet_temp` currently assumes uniform CPU, GPU, and Memory utilization across all racks within a DC for calculating power and outlet temperatures, scaling results from a single reference rack. This simplifies calculation but ignores potential load imbalance effects.
*   **IT Fan Control:** The IT fan speed responds to the *average* percentage utilization across CPU, GPU, and Memory, not necessarily the peak component temperature or load, which might differ from specific hardware implementations.
*   **Memory Power:** The current model only includes a static background power component based on total capacity and does not model dynamic power variations due to memory access intensity.
*   **HVAC Control:** CRAC setpoints are typically fixed via configuration; there is no active agent controlling HVAC setpoints in the default setup.
*   **Battery/Load Shifting:** While code structures exist (`bat_env`, `ls_env`), these components are generally inactive or fixed in the default SustainCluster configuration.

---

## 11. Planned Extensions (Beyond Current SustainCluster Implementation)

*   Integration of active RL control for battery energy storage.
*   Agent control over CRAC temperature setpoints.
*   More sophisticated airflow modeling (potentially simplified CFD or zonal models).
*   Dynamic memory power modeling.
