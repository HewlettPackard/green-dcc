
# SustainDC: Data Center Simulation Environment

This module implements a thermal and electrical model of a realistic datacenter to evaluate task scheduling policies under energy, carbon, and cost constraints.

It is integrated as a sub-environment of the global simulator and is compatible with Gym. It models internal racks, CPUs, airflow, cooling power, and battery interaction.

---

## 1. Overview

Each SustainDC instance models:
- **IT (Information Technology) load dynamics** (CPU-based task execution)
- **HVAC system** (cooling and chiller power)
- **Carbon & pricing** via CI and electricity managers

The datacenter consists of:
- `Racks` -> each holding multiple `CPUs`
- An `HVAC system` powered by CRAC units, chillers, cooling towers
- An internal thermal loop where internal temperature, fan velocity, and workload interact

---

## 2. Environment Structure

The main agent wrapper is `SustainDC`, which contains:
- `dc_env`: datacenter model (used for thermal + power simulation)
- `ls_env`: load-shifting simulator (currently bypassed)
- `bat_env`: battery simulator (currently fixed action)

Each step returns:
- IT power draw
- Cooling load (CRAC + chiller)
- Total power
- Water usage (via cooling tower)
- Airflow and temperature metrics

---

## 3. Resource Model

Resource units are simulated explicitly:
- `total_cpus`, `total_gpus`, and `total_mem` from `datacenters.yaml`
- Each task has CPU, GPU, and memory requirements
- Resource allocation is enforced at scheduling time
- GPU and memory modeling is **planned for upcoming extensions**

---

## 4. Power and Thermal Modeling

The datacenter model is built using a **bottom-up thermal-electrical modeling approach**:


### 4.1 CPU Power Model

Each CPU is modeled using a linear thermal power curve:

```math
P_{cpu} = \max(P_{idle}, P_{full} \cdot (\alpha_{cpu}(T_{in}) + \beta_{cpu} \cdot Load))
```

Where:
- `T_in`: inlet temperature
- `Load`: CPU utilization (0–1)
- `P_idle`, `P_full`: idle and max power (from config)
- `α`, `β`: fitted slope/intercept coefficients

---

### 4.2 Fan Power Model

```math
P_{fan} = P_{ref} \cdot \left(\frac{V_{fan}}{V_{ref}}\right)^3
```

Where:
- `V_fan`: computed airflow based on workload and inlet temperature

---

### 4.3 Rack Model

Each rack:
- Aggregates CPU and fan power
- Computes **outlet temperature** using airflow + load:

```math
T_{out} = T_{in} + c \cdot \frac{P_{IT}}{\rho \cdot C_{air} \cdot V_{fan}} + \text{offset}
```

---

### 4.4 IT Power Calculation

Total IT power per rack is the sum of CPU and fan power:

```math
P_{IT} = \sum_{r \in Racks} (P_{cpu}^{(r)} + P_{fan}^{(r)})
```

Where:
- `P_{cpu}^{(r)}`: power of each CPU in rack `r`
- `P_{fan}^{(r)}`: power of the fan in rack `r`
- `Racks`: list of racks in the datacenter
- `P_{IT}`: total IT power of the datacenter

---

## 5. HVAC System Modeling

The system models:

- **CRAC units**
- **Chillers**
- **Cooling towers**

### 5.1 Cooling Load:

```math
Q_{CRAC} = \rho \cdot C_{air} \cdot \dot{m}_{air} \cdot (T_{return} - T_{setpoint})
```

where `T_{return}` is the CRAC return air temperature, nd is estimated by averaging the outlet temperature of all racks, adjusted by a rack-specific return approach temperature:

```math
T_{return} = \frac{1}{N} \sum_{i=1}^{N} \left(T_{out}^{(i)} + \Delta T_{rack}^{(i)}\right)
```

Where:
- `T_out`: outlet air temperature of rack `i`
- `ΔT_rack`: return approach temperature (geometry-based offset)

This temperature is a key input to cooling load calculation.

---

### 5.2 CRAC Fan Power:

```math
P_{CRACfan} = P_{ref} \cdot \left(\frac{\dot{V}}{\dot{V}_{ref}}\right)^3
```

---

### 5.3 Chiller Power:

```math
P_{chiller} = f(COP, load, ambient)
```

---

## 6. Total Datacenter Power

The full datacenter power consists of:

```math
P_{total} = P_{IT} + P_{CRACfan} + P_{chiller} + P_{CTpump} + P_{CWpump}
```

Where:
- `P_IT`: sum of rack CPU and fan power
- `P_CRACfan`: CRAC air handling load
- `P_chiller`: calculated from thermal load and ambient
- `P_CTpump`, `P_CWpump`: cooling tower and chilled water pumps

These are dynamically computed at each timestep.

---

## 7. Water Usage Modeling

Water loss in the cooling tower is modeled as:

```math
W_{usage} = 0.044 \cdot T_{wetbulb} + (0.3528 \cdot \Delta T + 0.101)
```

---


## 8. Configurability

Each DC is defined in `datacenters.yaml` with:
- Compute resource spec: `total_cpus`, `total_gpus`, `total_mem`
- Power envelope via `datacenter_capacity_mw`
- Thermal config via `dc_config_file`

---

## 9. Model References

- [1] Sun et al., *Prototype Energy Models for Data Centers*, Energy & Buildings (2021)
- [2] Breen et al., *From chip to cooling tower data center modeling*, IEEE (2010)
- [3] Postema (2018), *Energy-efficient data centres*
- [4] NREL / EnergyPlus chiller specs (DOE Open Data)
- [5] SPX Cooling Tower Fan Efficiency Study (2020)

---

## 10. Planned Extensions

- GPU power modeling
- Memory energy modeling
- RL battery control
- CRAC setpoint agent
- Detailed airflow via CFD

---

## 11. Current Limitations

- GPU and memory models are not yet integrated into power model
- HVAC agent control is stubbed (fixed setpoint)
- Battery and load shifting agents are inactive by default
