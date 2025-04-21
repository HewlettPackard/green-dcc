# 🌍 GreenDCC: Multi-Data Center Geographical Scheduling Benchmark

## 1. Overview

**GreenDCC** is a high-fidelity simulation benchmark for **sustainable task scheduling** across globally distributed data center clusters. It enables research in carbon-aware, cost-efficient, and SLA-compliant workload scheduling using real-world datasets and realistic infrastructure models.

As AI workloads grow in scale and intensity, optimizing where and when these jobs run becomes critical, not just for performance, but for **energy cost**, **carbon emissions**, **cooling overheads**, and **network efficiency**.

GreenDCC captures this complexity by combining:
- Real AI Training and Inference workload traces (Alibaba Cluster Trace 2020)
- Region-specific carbon intensity, electricity prices, and weather data
- Transmission costs and emissions based on data size, cloud provider rates (AWS, Azure, GCP), and source-region carbon intensity
- SLA-aware task deferral and migration logic
- A reinforcement learning environment with customizable reward objectives

This benchmark is ideal for exploring new scheduling strategies across **sustainability**, **cost-efficiency**, and **operational reliability** at global scale. It includes a built-in **OpenAI Gym-compatible** environment and supports both **rule-based controllers** and **deep reinforcement learning agents**.


## 2. Quick Start

GreenDCC is ready to run out of the box with preconfigured training, simulation, and evaluation setups.

---

### 🔧 Installation

```bash
git clone https://github.com/YOUR_REPO/green-dcc.git
cd green-dcc
conda create -n green-dcc python=3.10
conda activate green-dcc
pip install -r requirements.txt
```

---

### 🚀 Train the RL Agent (SAC)

The environment uses a Soft Actor-Critic (SAC) agent implemented in [`train_rl_agent.py`](train_rl_agent.py).

Start training with:

```bash
python train_rl_agent.py
```

To use custom configurations and track your run:

```bash
python train_rl_agent.py \
  --sim-config configs/env/sim_config.yaml \
  --dc-config configs/env/datacenters.yaml \
  --reward-config configs/env/reward_config.yaml \
  --algo-config configs/env/algorithm_config.yaml \
  --tag my_experiment
```

This will:
- Load the Alibaba-based workload and sustainability datasets
- Create a 7-day simulation window with 5 global datacenters
- Launch training with your reward objectives and RL hyperparameters

---

### 📈 Monitor Training with TensorBoard

```bash
tensorboard --logdir runs/
```

Then open your browser at:
```
http://localhost:6006
```

---

### 🤖 Rule-Based Baselines

You can also run **predefined controllers** from [`utils/task_assignment_strategies.py`](utils/task_assignment_strategies.py):

- `distribute_random`
- `distribute_round_robin`
- `distribute_most_available`
- `distribute_least_pending`
- `distribute_lowest_price`
- `distribute_lowest_carbon`
- `distribute_lowest_network_cost`
- `distribute_weighted` (custom scoring function)

---

### 🧪 Evaluate Agents and Baselines

Use the notebook [`eval_agent_notebook.py`](eval_agent_notebook.py) to:

- Load a trained RL checkpoint
- Run a 7-day simulation
- Compare against rule-based policies
- Plot performance metrics (energy, carbon, SLA)

> ✅ A ready-to-run Google Colab notebook is also available for testing:
> [Open in Colab](https://colab.research.google.com/drive/1LLw313sG56l2I29E0Q9zh6KM0q5Z23WX?usp=sharing)

---

### ⚙️ Configuration Files

| File                                      | Description                                  |
|-------------------------------------------|----------------------------------------------|
| `configs/env/sim_config.yaml`             | Simulation time window, workload source      |
| `configs/env/datacenters.yaml`            | DC locations, capacities, timezones          |
| `configs/env/reward_config.yaml`          | Composite reward function setup              |
| `configs/env/algorithm_config.yaml`       | SAC training parameters and buffer setup     |

All configs are editable and fully modular.

---

## 3. Key Features

GreenDCC combines real-world data and high-fidelity infrastructure modeling to simulate global workload scheduling with sustainability objectives.

---

### 🌍 Global Coverage

- Supports 20+ real-world datacenter regions using standardized location codes (e.g., `"US-CAL-CISO"`, `"DE-LU"`, `"BR-SP"`, `"SG"`, `"AU-NW"`, etc.)
- Integrates **real energy prices**, **carbon intensity**, and **weather** data for each region

---

### 🧠 DRL + Rule-Based Control

- Built-in **Soft Actor-Critic (SAC)** agent for training policies via RL ([`train_rl_agent.py`](train_rl_agent.py))
- Supports **time defer or schedule** actions for every task
- Modular reward engine with **multi-objective** functions
- Compare RL agents against **rule-based policies** like:
  - Most available
  - Lowest price
  - Lowest carbon
  - Round-robin
  - Weighted heuristics

---

### ⚡ Sustainability-Aware Simulation

- Realistic modeling of:
  - Energy use (CPU/GPU/Memory)
  - Cooling dynamics based on weather and workload
  - Carbon emissions from compute + data transmission
  - SLA penalties for missed deadlines
- Full energy and cooling model in `envs/sustaindc/`, combining formulas from EnergyPlus and scientific literature.
📄 References and modeling details are available in [`envs/sustaindc/README_SustainDC.md`](envs/sustaindc/README_SustainDC.md)



---

### 🔁 Time & SLA-Aware Scheduling

- Simulation runs at 15-minute intervals
- Each task has an SLA deadline:
  - Immediate, Normal, or Flexible
- Agent can **defer tasks** for better sustainability outcomes
- Deferral incurs SLA penalties if missed

---

### 🔬 Modular Reward System

Located in [`rewards/predefined/`](rewards/predefined/):

- `energy_price_reward.py`
- `carbon_emissions_reward.py`
- `transmission_cost_reward.py`
- `transmission_emissions_reward.py`
- `sla_penalty_reward.py`
- `efficiency_reward.py`
- `composite_reward.py` — combine any subset with weights

---

### 🧪 Evaluation & Visualization

- Evaluate agents using [`eval_agent_notebook.py`](eval_agent_notebook.py)
- Compare SAC vs rule-based baselines
- Log training performance to TensorBoard
- Built-in plots for:
  - Energy/cost/carbon trends
  - Resource usage
  - Task distributions

---

### 🔧 Config-Driven and Extensible

- All components configurable via YAML files:
  - `datacenters.yaml`, `reward_config.yaml`, `sim_config.yaml`, etc.
- Add new rule-based or RL controllers by extending:
  - [`utils/task_assignment_strategies.py`](utils/task_assignment_strategies.py)
  - [`rl_components/`](rl_components/)


## 4. Benchmark Architecture

GreenDCC simulates a real-world scenario where AI workloads are generated across the globe and scheduled by a central agent under sustainability and cost constraints.

---

### 🧭 Global Workflow

At every 15-minute timestep:

1. **Task Generation**
   - Tasks are generated at origin datacenters using population × time-of-day logic.
   - Based on filtered and normalized traces from Alibaba 2020.

2. **Global Scheduling**
   - A centralized controller (RL or rule-based) selects where to execute each task.
   - It can also defer execution to future timesteps (action = 0).

3. **Routing & Transfer**
   - If the selected destination ≠ origin, data is transferred with:
     - Monetary transmission cost (cloud-provider rates)
     - Estimated energy and carbon overhead

4. **Execution**
   - Tasks are queued at destination datacenter and executed when resources are available.
   - Simulation tracks energy use, cooling needs, carbon footprint, and SLA status.

---

### 🔁 15-Minute Timestep Justification

GreenDCC simulates **thermal and power dynamics**, which change slowly and require coarser control cycles:

- Energy price, carbon intensity, and weather data is sampled at 15-min intervals.
- HVAC systems and thermal buffers operate on similar timescales.
- Cited studies include:
  - DeepEE (ICDCS 2019) [https://doi.org/10.1109/ICDCS.2019.00070](https://doi.org/10.1109/ICDCS.2019.00070)
  - Green RL Cooling (ACM TCPS 2023) [https://doi.org/10.1145/3582577](https://doi.org/10.1145/3582577)
  - EnergyPlus models [link](https://dmey.github.io/EnergyPlusFortran-Reference/proc/calcelectricchillermodel.html)

---

### 🧠 System Components

```text
            +-----------------------+
            |  TaskSchedulingEnv    |  ← Gym wrapper for RL agents
            +-----------------------+
                         ↓
     +---------------------------------------------+
     |      DatacenterClusterManager               |  ← Routes tasks, tracks metrics
     |  +----------------+    +----------------+   |
     |  | SustainDC (DC1)|    | SustainDC (DC2)|...|
     |  +----------------+    +----------------+   |
     +---------------------------------------------+
```

- `TaskSchedulingEnv`: Interfaces with RL agents; processes actions; builds reward.
- `DatacenterClusterManager`: Manages DC state and task routing.
- `SustainDC`: Simulates energy, carbon, cooling, and scheduling dynamics per DC.

---

### 🌐 Centralized Scheduling, Decentralized Execution

- **Central agent** makes placement decisions based on global view.
- Each **datacenter simulates local execution**, thermal constraints, and SLA outcomes.
- Observations are made per task, including:
  - Time features (hour, day)
  - Resource availability at each DC
  - Prices, carbon intensities, pending queues
  - SLA deadlines

---


## 5. Datasets

GreenDCC combines multiple real-world datasets to simulate the sustainability and operational environment of distributed data centers.

---

### 5.1 📦 Workload Dataset: Alibaba Cluster Trace 2020

We use the [Alibaba Cluster Trace 2020](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020), which contains training and inference jobs from over 6,500 GPUs.

Key processing steps:
- Filtered to include only jobs ≥ 15 minutes
- Extended to 1 year by replicating daily patterns
- Jobs grouped into 15-minute intervals
- Origin DCs assigned using population × time-of-day logic
- Bandwidth estimated per task to simulate transfer cost and delay

Processed files are stored as `.pkl` in:
```
data/workload/alibaba_2020_dataset/
```

📄 See: [`data/workload/README.md`](data/workload/README.md)

---

### 5.2 ⚡ Sustainability Datasets

GreenDCC includes time-aligned data from 2021–2024 for each region:

| Type             | Source                                   |
|------------------|------------------------------------------|
| Weather          | [Open-Meteo](https://open-meteo.com/)    |
| Carbon Intensity | [Electricity Maps](https://electricitymaps.com/) |
| Electricity Price| [GridStatus](https://gridstatus.io/), CAISO, OMIE, etc.

Data is normalized to:
- UTC timestamps
- USD/kWh or gCO₂/kWh units
- Available under `data/electricity_prices/`, `data/weather/`, `data/carbon_intensity/`

📄 See: [`data/electricity_prices/README.md`](data/electricity_prices/README.md) and  
📄 [`data/carbon_intensity/Carbon_Intensity_Data_Guide.md`](data/carbon_intensity/Carbon_Intensity_Data_Guide.md)

---

### 5.3 🌐 Transmission Cost and Emissions

Task migration across DCs incurs transfer cost and transmission emissions.

- Transfer **cost per GB** sourced from:
  - [AWS](https://aws.amazon.com/ec2/pricing/on-demand/)
  - [GCP](https://cloud.google.com/vpc/pricing)
  - [Azure](https://azure.microsoft.com/en-us/pricing/details/bandwidth/)
- Transmission **energy intensity** is set to `0.06 kWh/GB`
- Associated carbon emissions calculated as:
```python
emissions_kg = task.bandwidth_gb × 0.06 × carbon_intensity_origin
```

Matrix files and calculators are in:
```
utils/transmission_cost_loader.py
utils/transmission_region_mapper.py
```

---

All datasets are pre-aligned for 15-minute simulation steps and used to drive dynamic price, carbon, and sustainability behavior during scheduling.

---


## 6. Simulation Mechanics

GreenDCC simulates a global scheduling problem in 15-minute intervals, where each action affects sustainability and operational outcomes.

---

### 6.1 ⏱️ Timestep and Time Resolution

- Each simulation step represents **15 minutes**
- Matches resolution of:
  - Electricity Maps (carbon intensity)
  - GridStatus (energy price)
  - Open-Meteo (weather)
- Thermal and electrical systems respond on similar timescales
- Enables stable control, resource allocation, and HVAC planning

---

### 6.2 🧠 Action Space

Each agent decision is per-task:
```python
action ∈ {0, 1, ..., N}
```
- `0` → Defer task (try again next step)
- `1..N` → Assign task to datacenter `i`

This allows **temporal deferral** when sustainability conditions are unfavorable.

---

### 6.3 🕒 SLA Modeling

Each task has a deadline:
```python
sla_deadline = arrival_time + sla_multiplier * duration
```

SLA tiers:
| Priority | SLA Multiplier | Meaning                      |
|----------|----------------|------------------------------|
| Urgent   | 1.1            | Must run immediately         |
| Normal   | 1.5            | Can be deferred slightly     |
| Flexible | 2.0            | Can wait for optimal moment  |

If a task misses its SLA deadline, it is counted as a **violation** and penalized.

---

### 6.4 🎯 Reward Function

GreenDCC uses a modular reward engine:
- Single or multi-objective
- Composite weighting
- Available reward components:
  - `energy_price`
  - `carbon_emissions`
  - `energy_consumption`
  - `transmission_cost`
  - `transmission_emissions`
  - `sla_penalty`
  - `efficiency`

Example:
```yaml
reward:
  components:
    energy_price:
      weight: 0.5
    carbon_emissions:
      weight: 0.3
    sla_penalty:
      weight: 0.2
```

🗂️ Location: [`rewards/predefined/`](rewards/predefined/)

---

### 6.5 🏭 Datacenter Modeling

Each datacenter (`SustainDC`) models:

- CPU, GPU, and Memory availability
- Task queueing and scheduling
- Weather-aware cooling demand
- Carbon and energy usage tracking
- Water usage (optional)
- Resource constraints and drop behavior

Energy and cooling models use:
- Equations from **EnergyPlus**
- Methods from recent datacenter energy research
- Full documentation: [`envs/sustaindc/README_SustainDC.md`](envs/sustaindc/README_SustainDC.md)

---

All simulation components are plug-and-play and compatible with OpenAI Gym via `TaskSchedulingEnv`.

---

## 7. Advanced Training and Configuration

This section explains the internal structure of GreenDCC’s training pipeline and how to fully customize it for your experiments.

---

### 7.1 ⚙️ Configuration Files Explained

GreenDCC separates simulation, reward, and algorithm configs for modularity. All config files are located in `configs/env/`.

#### 🧩 `sim_config.yaml`

Defines:
- Simulation start time (`year`, `month`, `init_day`, `init_hour`)
- Duration in days (`duration_days`)
- Task trace file (`workload_path`)
- Cloud provider (`gcp`, `aws`, `azure`) for transfer pricing
- Strategy: `"manual_rl"` for RL agent, or rule-based alternatives
- TensorBoard and log output control

#### 🧩 `datacenters.yaml`

Each entry describes one datacenter:
- `dc_id`: Unique numeric ID
- `location`: Region code (e.g., `"DE-LU"`)
- `timezone_shift`: Hours offset from UTC
- `population_weight`: Used in origin generation
- `total_cores`, `total_gpus`, `total_mem`: Available compute resources
- `dc_config_file`: JSON with HVAC and layout modeling info

#### 🧩 `reward_config.yaml`

Defines your optimization objective:
- Set `normalize: true/false` if you want to normalize each component in the reward function
- Add reward components with:
  - `weight`: importance in the total reward
  - `args`: component-specific parameters

```yaml
reward:
  normalize: false
  components:
    energy_price:
      weight: 0.5
      args: { normalize_factor: 100000 }
    carbon_emissions:
      weight: 0.3
      args: { normalize_factor: 10 }
    sla_penalty:
      weight: 0.2
      args: { penalty_per_violation: 5.0 }
```

#### 🧩 `algorithm_config.yaml`

Defines SAC hyperparameters:
- `alpha`, `gamma`, `batch_size`, `tau`
- `total_steps`: number of training steps
- `warmup_steps`: random policy before learning
- `policy_update_frequency`, `save_interval`

---

### 7.2 ⚙️ Training Execution Flow

When you run `train_rl_agent.py`:

1. **Load YAML configs** with `load_yaml()`
2. **Build the environment** with:
   - `DatacenterClusterManager` (tracks task state)
   - `TaskSchedulingEnv` (Gym-compatible wrapper)
3. **Initialize SAC agent**:
   - Actor, Critic, Target Critic
   - Replay buffer, optimizers
4. **Main loop**:
   - Sample actions from actor
   - Step the environment
   - Store experience
   - Periodically update actor/critic
5. **Log to TensorBoard**:
   - Rewards, losses, entropy
6. **Save checkpoints** every `save_interval` steps

---

### 7.3 🎯 Custom and Composite Reward System

All reward components are in:
📂 [`rewards/predefined/`](rewards/predefined/)

Built-in components include:
- `energy_price_reward.py`
- `carbon_emissions_reward.py`
- `sla_penalty_reward.py`
- `transmission_cost_reward.py`
- `transmission_emissions_reward.py`
- `efficiency_reward.py`

To create your own reward:
1. Create a file like `my_custom_reward.py` under `rewards/predefined/`
2. Inherit from `BaseReward`
3. Implement `__call__()` and optionally `get_last_value()`

For more information, see: [`rewards/README.md`](rewards/README.md)

Example:
```python
class MyCustomReward(BaseReward):
    def __call__(self, cluster_info, current_tasks, current_time):
        value = ... # compute custom metric
        return -value
```

Then register it in `reward_config.yaml`:
```yaml
reward:
  components:
    my_custom_reward:
      weight: 0.2
      args: {}
```

---

### 7.4 🧠 Extending to New Agents

GreenDCC is agent-agnostic — it only requires:
- Per-task discrete actions: `action ∈ {0, ..., N}`
- OpenAI Gym-compatible logic (`reset`, `step`, `observation_space`, `action_space`)

To add PPO or other agents:
- Use `TaskSchedulingEnv` from `envs/task_scheduling_env.py`
- Step with:
```python
obs, reward, done, truncated, info = env.step(actions)
```
- All observations are per-task, with task + system-level features

RL components (networks, buffers) are modular in:
📂 `rl_components/`

---

GreenDCC is designed for flexibility. You can define your own:
- Tasks and resource profiles
- Optimization objectives
- Scheduling strategies
- Reward logic
- Agent architectures

---

## 8. Evaluation and Visualization

GreenDCC includes a powerful evaluation pipeline that allows you to simulate a 7-day workload and generate metrics and visualizations comparing RL agents and rule-based baselines.

---

### 📊 Evaluation Flow

- Evaluation runs are controlled via [`eval_agent_notebook.py`](eval_agent_notebook.py) or Python scripts.
- You can switch between an RL agent (`manual_rl`) or a rule-based controller by modifying the `strategy` field.
- The script evaluates:
  - Total energy cost and carbon emissions
  - SLA violation rate
  - Transmission cost and delays
  - Resource utilization (CPU, GPU, Memory)

---

### 📈 Summary Statistics

Evaluation generates a summary per datacenter including:

| Metric                    | Description                          |
|---------------------------|--------------------------------------|
| Total Energy Cost (USD)   | Sum of electricity cost              |
| Total Energy (kWh)        | Total energy consumed                |
| Total CO₂ (kg)            | Total carbon emissions               |
| SLA Violation Rate (%)    | Percentage of jobs violating SLA     |
| Avg Utilization (%)       | CPU, GPU, Memory over 7 days         |
| Avg External Temperature  | Used for cooling simulation          |
| Avg Carbon Intensity      | gCO₂ per kWh from electricity source |

---

### 📉 Built-in Plots

The following plots are generated by default:

- Energy price over time per datacenter
- Tasks assigned per datacenter
- Total running tasks
- Carbon intensity trends
- Transmission cost and delayed tasks
- CPU, GPU, and Memory utilization
- External temperature variations

These help visualize performance trade-offs and sustainability behavior.

---

### 🧪 Evaluation Metrics Source

The evaluation script logs information from:

- `info["datacenter_infos"]`: carbon, cost, SLA, utilization
- `info["transmission_cost_total_usd"]`: inter-DC bandwidth cost
- `dc_cpu_workload_fraction`: estimated workload pressure

All collected into a `DataFrame`, summarized and plotted using seaborn and matplotlib.

---

### ✅ Output Summary

You can generate a tabular summary:

```python
summary = df.groupby("datacenter").agg({
    "energy_cost": "sum",
    "energy_kwh": "sum",
    ...
}).reset_index()
```

This allows precise, datacenter-level benchmarking.

---

To use your own agent checkpoint, set:
```python
checkpoint_path = "checkpoints/train_<date>/best_checkpoint.pth"
```

To switch to rule-based baselines, change:
```python
strategy = "lowest_price"  # or "random", "most_available", etc.
```

Full control is available in the code — extend or modify as needed.

---

## 9. How to Contribute

GreenDCC is designed to be modular and extensible. We welcome contributions from the community to expand the dataset, models, or evaluation tools.

---

### 🌍 Adding a New Datacenter Location (with Full Dataset Integration)

To simulate a new region accurately, you need to provide aligned data for:
- **Electricity Prices**
- **Carbon Intensity**
- **Weather (Temperature)**
- **Transmission Costs** (optional, but recommended)

#### 📁 Step-by-Step:

1. **Update `datacenters.yaml`**
   - Add your region with a unique `dc_id`
   - Set `location`, `timezone_shift`, `population_weight`, and hardware specs
   - Example:
     ```yaml
     - dc_id: 6
       location: "FI"  # Finland
       timezone_shift: 2
       population_weight: 0.12
       total_cores: 4000
       total_gpus: 500
       total_mem: 4000
       dc_config_file: "configs/dcs/dc_config.json"
     ```

2. **Electricity Prices**
   - Save as: `data/electricity_prices/standardized/FI/YEAR/FI_electricity_prices_YEAR.csv`
   - Format: `timestamp_utc, price_usd_per_mwh`
   - Use extraction script: `data/electricity_prices/scripts/`

3. **Carbon Intensity**
   - Save as: `data/carbon_intensity/standardized/FI_YEAR.csv`
   - Format: `timestamp_utc, carbon_intensity_gco2_per_kwh`
   - Follow structure in: `data/carbon_intensity/analyze_carbon_intensity_data.py`

4. **Weather (Temperature)**
   - Save as: `data/weather/standardized/FI/YEAR/FI_weather.csv`
   - Format: `timestamp_utc, temperature_celsius`
   - Use helper: `data/weather/extract_weather_data.py`

5. **Transmission Costs**
   - Update matrix in `utils/transmission_cost_loader.py`
   - Add row and column for your region with cost per GB (USD)

6. **Mapping (Optional)**
   - If using GCP/AWS mapping logic, update: `utils/transmission_region_mapper.py`

Once added, your new datacenter will be treated as any other by the simulator, agent, and all metrics pipelines.

You need to obtain the data from the simulated year (2021-2024) for your region.

For a working example, inspect how "DE-LU" or "SG" is handled throughout the system.

---

### 🔧 Add a New Datacenter

To define a new datacenter:
1. Add its entry to `configs/env/datacenters.yaml`
2. Assign a unique `dc_id`, location code, and resource specs
3. If needed, generate energy/carbon/weather data for that location
4. Set `dc_config_file` to your HVAC and layout configuration (JSON)

---

### 🤖 Add a New Rule-Based Controller

1. Implement your function in `utils/task_assignment_strategies.py`
```python
def distribute_my_policy(task, datacenters, logger):
    # Choose a datacenter based on your logic
    return selected_dc_id
```

2. Add it to the strategy map inside the simulation loop.

---

### 🧠 Add a New RL Agent

GreenDCC uses Gym-compatible environments. You can plug in:
- PPO, A2C, TD3, or any PyTorch-based policy
- Use `TaskSchedulingEnv` as your environment entrypoint
- Access observations, actions, rewards, and info dicts

---

### 🧪 Add Visualizations

- Extend `eval_agent_notebook.py` or your custom scripts
- Use the full datacenter metrics from the info dictionary
- Plot carbon, SLA, utilization, and more with seaborn/matplotlib

---

Open a PR or issue with your proposal. We're happy to collaborate!

---

## 10. License & Citation

GreenDCC is released under the **MIT License**.  
Attribution to original data sources is required.

---

### 📚 Citing This Project

If you use GreenDCC in your work, please cite the benchmark and the original datasets:

#### GreenDCC (Benchmark Paper – WIP)
```
@misc{greendcc2025,
  title={GreenDCC},
  author={WIP},
  year={2025},
  url={https://github.com/YOUR_REPO/green-dcc},
  note={Preprint}
}
```

#### Alibaba Cluster Trace 2020
```
@inproceedings{weng2022mlaas,
  title={MLaaS in the Wild: Workload Analysis and Scheduling in Large-Scale Heterogeneous {GPU} Clusters},
  author={Weng, Tao and others},
  booktitle={NSDI},
  year={2022}
}
```

---

### 📦 Data Sources

- [Alibaba Cluster Trace](https://github.com/alibaba/clusterdata/)
- [Electricity Maps](https://www.electricitymaps.com/)
- [Open-Meteo](https://open-meteo.com/)
- [GridStatus.io](https://gridstatus.io/)
- [AWS](https://aws.amazon.com/ec2/pricing/on-demand/), [GCP](https://cloud.google.com/vpc/pricing), [Azure](https://azure.microsoft.com/en-us/pricing/details/bandwidth/)

We thank all providers for making these resources openly available.