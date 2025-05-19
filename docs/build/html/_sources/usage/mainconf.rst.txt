.. _mainconf_ref:

Main Configuration File
=======================

Description of the three top-level configuration objects used by the environment loader:

- **data_center_configuration** (object)  
- **hvac_configuration** (object)  
- **server_characteristics** (object)  

Field Reference
---------------

data_center_configuration (object)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**NUM_ROWS** (integer)  
    Number of rack rows in each hall.

**NUM_RACKS_PER_ROW** (integer)  
    Number of racks in each row.

**RACK_SUPPLY_APPROACH_TEMP_LIST** (list of floats)  
    Supply-air approach temperatures (°C) for each rack; list length must equal  
    NUM_ROWS × NUM_RACKS_PER_ROW.

**RACK_RETURN_APPROACH_TEMP_LIST** (list of floats)  
    Return-air approach temperatures (°C) for each rack; same length and order  
    as the supply list.

**CPUS_PER_RACK** (integer)  
    Number of CPU cores per rack.

**GPUS_PER_RACK** (integer)  
    Number of GPU devices per rack.

hvac_configuration (object)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**C_AIR** (float)  
    Specific heat capacity of air (J·kg⁻¹·K⁻¹).

**RHO_AIR** (float)  
    Air density (kg·m⁻³).

**CRAC_SUPPLY_AIR_FLOW_RATE_pu** (float)  
    CRAC unit supply air flow rate (per unit).

**CRAC_REFRENCE_AIR_FLOW_RATE_pu** (float)  
    Reference supply air flow rate for CRAC (per unit).

**CRAC_FAN_REF_P** (float)  
    Reference fan power for CRAC (W).

**CHILLER_COP_BASE** (float)  
    Base coefficient of performance for the chiller.

**CHILLER_COP_K** (float)  
    Temperature coefficient for chiller COP.

**CHILLER_COP_T_NOMINAL** (float)  
    Nominal temperature (°C) for COP calculation.

**CT_FAN_REF_P** (float)  
    Reference fan power for cooling tower (W).

**CT_REFRENCE_AIR_FLOW_RATE** (float)  
    Reference air flow rate for cooling tower (m³·s⁻¹).

**CW_PRESSURE_DROP** (float)  
    Pressure drop in condenser water loop (Pa).

**CW_WATER_FLOW_RATE** (float)  
    Water flow rate in condenser water loop (m³·s⁻¹).

**CW_PUMP_EFFICIENCY** (float)  
    Pump efficiency of condenser water loop.

**CT_PRESSURE_DROP** (float)  
    Pressure drop in cooling tower water loop (Pa).

**CT_WATER_FLOW_RATE** (float)  
    Water flow rate in cooling tower loop (m³·s⁻¹).

**CT_PUMP_EFFICIENCY** (float)  
    Pump efficiency of cooling tower loop.

server_characteristics (object)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Controls power and airflow profiles for servers (CPU and GPU). Defaults and fallbacks are described below.

CPU Profile
------------

The system's default CPU profiles are [130 W, 10 W], [110 W, 10 W] and [170 W, 10 W]. If the  
`DEFAULT_CPU_POWER_CHARACTERISTICS` array is omitted entirely, the loader  
falls back to the HP PROLIANT profile.

GPU Profiles
------------

The system’s default GPU profiles are P100 and A6000. If the  
`DEFAULT_GPU_POWER_CHARACTERISTICS` array is omitted entirely, the loader  
falls back to the V100 profile.

**NVIDIA_P100** (list of two integers)  
    Maximum power consumption: 250 W [1]_  
    Idle power consumption: 25 W [2]_

**NVIDIA_A6000** (list of two integers)  
    Maximum power consumption: 250 W [3]_  
    Idle power consumption: 22 W [4]_

Fallback Profile
----------------

**NVIDIA_V100** (list of two integers)  
    Maximum power consumption: 300 W [5]_  
    Idle power consumption: 70 W [6]_

**HP_PROLIANT** (list of two integers)  
    Maximum power consumption: 170 W [5]_  
    Idle power consumption: 110 W [6]_

References
----------

.. [1] NVIDIA Corporation (2016) *NVIDIA Tesla P100 PCIe Data Sheet* [PDF]. Available at: https://images.nvidia.com/content/tesla/pdf/nvidia-tesla-p100-PCIe-datasheet.pdf

.. [2] Ali, G., Bhalachandra, S., Wright, N., Sill, A. and Che, Y. (2020) *Evaluation of power counters and controls on general-purpose GPUs*. SC20: International Conference for High Performance Computing, Networking, Storage and Analysis (SC ’20), Poster. Available at: https://sc20.supercomputing.org/proceedings/tech_poster/poster_files/rpost131s2-file2.pdf

.. [3] NVIDIA Corporation (2023) *vGPU A16 Data Center Solutions Data Sheet* [PDF]. Available at: https://images.nvidia.com/content/Solutions/data-center/vgpu-a16-datasheet.pdf

.. [4] Khandelwal, S., Wadhwa, E. and Shreejith, S. (2022) ‘Deep Learning-based Embedded Intrusion Detection System for Automotive CAN’, in *Proceedings of the IEEE 33rd International Conference on Application-specific Systems, Architectures and Processors (ASAP)*, Gothenburg, Sweden, pp. 88–92. doi: 10.1109/ASAP54787.2022.00023.  

.. [5] NVIDIA Corporation (2018) *Tesla V100 Data Sheet* [PDF]. Available at: https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet-letter-fnl-web.pdf

.. [6] You, J., Chung, J.-W. and Chowdhury, M. (2023) ‘Zeus: Understanding and Optimizing GPU Energy Consumption of DNN Training’, in Balakrishnan, M. and Ghobadi, M. (eds.) *Proceedings of the 20th USENIX Symposium on Networked Systems Design and Implementation (NSDI 2023)*, Boston, MA, April 17–19, 2023. USENIX Association, pp. 119–139. Available at: https://www.usenix.org/conference/nsdi23/presentation/you (Accessed: [date]).
