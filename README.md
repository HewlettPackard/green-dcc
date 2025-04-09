
# Multi-Data Center Geographical Scheduling Benchmark

A high-fidelity simulation benchmark for **sustainable task scheduling** across globally distributed data centers. Optimize AI workloads based on **carbon emissions**, **energy cost**, **resource efficiency**, **transmission costs**, and **SLA guarantees**.

> **Goal**: Assign each incoming task to the best datacenter considering real-world sustainability and operational trade-offs.

---

## Features

- **Centralized global scheduler** with decentralized task generation
- Real-world data from Alibaba, Electricity Maps, Open-Meteo, AWS/GCP/Azure
- Transfer-aware routing (latency, bandwidth, transmission cost)
- Detailed simulation of:
  - Energy use
  - Carbon emissions
  - Cooling (temperature-based proxy)
  - Transmission overheads
- Supports:
  - Rule-based controllers
  - Deep RL agents (SAC pre-implemented)
- Modular reward function system
- RL-ready Gym-compatible environments
- Fully extensible and interpretable

---

## Benchmark Design

At every 15-minute timestep:

1. Task Generation: 
   - Tasks are generated in multiple datacenters using population and time-zone-aware logic.
2. Global Scheduling:
   - A centralized agent observes system-wide state and decides a destination DC for each task.
3. Routing Penalty:
   - Tasks sent to remote DCs incur transfer costs and delays.
4. Execution:
   - Each DC executes tasks when resources are available.
   - Energy, cost, and carbon metrics are recorded.

---

## Supported Optimization Objectives

- Total energy cost (USD)
- Carbon emissions (kg CO₂)
- Energy consumption (kWh)
- SLA violations
- Inter-datacenter transmission costs
- Multi-objective trade-offs (via composite rewards)

---

## Real-World Datasets

| Type               | Source                                                                 |
|--------------------|------------------------------------------------------------------------|
| AI Workloads       | Alibaba Cluster Trace 2020                                             |
| Weather Data       | Open-Meteo                                                             |
| Carbon Intensity   | Electricity Maps                                                       |
| Energy Prices      | Mixed sources: Electricity Maps, GridStatus, Open APIs (per country)   |
| Transmission Costs | AWS, GCP, Azure (per region)                                           |

---

## Dataset Format

The cleaned dataset is saved as a Pandas .pkl file with the following structure:

interval_15m | tasks_matrix  
-------------|----------------------------------------  
2020-03-01   | [[job1, tstart, tend, cpu, gpu, mem, bw], ...]

Each task:
- Normalized CPU, GPU, MEM usage
- Bandwidth (GB)
- Dynamically assigned origin DC using local time + population logic

---

## Architecture

```
           +------------------+ 
           | Global Scheduler |
           +------------------+ 
                    ↑  
               [All Tasks]  
                    ↑  
+----------+   +----------+   +----------+  
|   DC1    |   |   DC2    |   |   DC3    |   ...
+----------+   +----------+   +----------+ 
```
Each DC: local environment with
- Resource tracking
- Energy & carbon simulation
- Scheduling queue

---

## Supported Locations

These are valid `location` codes to use when defining datacenters in your simulation:

| Code         | Region / Market                          |
|--------------|------------------------------------------|
| US-NY-NYIS   | New York (NYISO)                         |
| US-CAL-CISO  | California (CAISO)                       |
| US-TEX-ERCO  | Texas (ERCOT)                            |
| DE-LU        | Germany + Luxembourg (ENTSO-E)           |
| FR           | France (ENTSO-E)                         |
| SG           | Singapore (USEP)                         |
| JP-TK        | Japan - Tokyo Area (JEPX)                |
| IN           | India (POSOCO)                           |
| AU-NSW       | Australia - New South Wales (AEMO)       |
| BR           | Brazil (ONS)                             |
| ZA           | South Africa (Eskom)                     |
| PT           | Portugal (OMIE)                          |
| ES           | Spain (OMIE)                             |
| BE           | Belgium (ENTSO-E)                        |
| CH           | Switzerland (ENTSO-E)                    |
| KR           | South Korea (KPX)                        |
| CA-ON        | Ontario (IESO)                           |
| CL-SIC       | Chile (CDEC-SIC)                         |
| AT           | Austria (ENTSO-E)                        |
| NL           | Netherlands (ENTSO-E)                    |

---

## 📁 Code Structure

```
envs/                         # RL-compatible Gym environments
├── env_config.py            # Config class for environments
├── task_scheduling_env.py   # Global Gym wrapper for training & evaluation
├── sustaindc/               # Internal simulation for datacenter agents
│   ├── __init__.py
│   ├── sustaindc_env.py     # Main multi-agent SustainDC environment
│   ├── battery_env.py       # Battery simulation (forward battery model)
│   ├── battery_model.py     # Battery power & capacity dynamics
│   ├── timeloadshifting_env.py # Load shifting queue & SLA simulation
│   ├── datacenter_model.py  # Physical data center IT & HVAC model
│   └── dc_gym.py            # Gym interface for the datacenter model
```

```
simulation/                  # High-level simulator
├── __init__.py
└── datacenter_cluster_manager.py   # Manages multiple datacenters + task routing
```

```
rl_components/               # RL agent logic and training utilities
├── agent_net.py             # Actor neural network (for SAC or other RL)
├── replay_buffer.py         # Experience replay buffer
└── task.py                  # Task class (job ID, resource needs, SLA, etc.)
```

```
rewards/                     # Reward function system
├── base_reward.py           # Reward interface
├── reward_registry.py       # Auto-loading and registry
└── predefined/              # All predefined rewards
    ├── energy_price_reward.py       # Energy price reward
    ├── carbon_emissions_reward.py   # Carbon emissions reward
    ├── energy_consumption_reward.py # Energy consumption reward
    ├── sla_penalty_reward.py        # SLA penalty reward
    ├── efficiency_reward.py         # Efficiency reward
    └── composite_reward.py          # Composite reward for multiple objectives

```
```
utils/                       # Utilities and managers
├── make_envs_pyenv.py       # Functions to construct internal envs
├── managers.py              # CI, price, time, weather managers
├── utils_cf.py              # Config and helper utilities
├── dc_config_reader.py      # Parse & process DC config files
└── task_assignment_strategies.py  # Rule-based task routing policies
```

---

## Example Use Cases

- RL agent for low-cost, low-carbon scheduling
- Rule-based heuristics (most available, least emissions, etc.)
- Evaluate multi-objective optimization strategies
- Ablation on reward signals (carbon vs. cost vs. transfer)

---

## Example Metrics Output

Datacenter DC1:
   - Total Tasks Executed: 14640
   - Total Energy Used: 285,481.07 kWh
   - Average CPU Usage: 58.99%
   - Total Bandwidth Used: 58,818 GB

Datacenter DC2:
   - Task Share: 18.95%
   - Avg Energy per Step: 275.63 kWh

---

## Modular Reward System

GreenDCC comes with a powerful modular reward engine. You can:

- Optimize for **single** or **multiple** objectives
- Use built-in rewards like `energy_price`, `carbon_emissions`, `transmission_cost`
- Create **custom rewards** easily
- Combine multiple rewards with weights using `CompositeReward`


## Built-in Reward Functions

The following reward components are already available:

| Reward Name        | Description                                      | Params                     |
|--------------------|--------------------------------------------------|----------------------------|
| `energy_price`     | Penalizes total energy cost of tasks             | `normalize_factor`         |
| `carbon_emissions` | Penalizes total kgCO₂ emissions                  | `normalize_factor`         |
| `energy_consumption` | Penalizes total energy used (in kWh)          | `normalize_factor`         |
| `efficiency`       | Penalizes energy per scheduled task              | _None_                     |
| `sla_penalty`      | Penalizes number of SLA violations               | `penalty_per_violation`    |
| `composite`        | Combines multiple reward components              | See below                  |


👉 [See full reward documentation here »](rewards/README.md)

---

## 📊 Example Composite Reward

```python
from rewards.predefined.composite_reward import CompositeReward

reward_fn = CompositeReward(
    components={
        "energy_price": {
            "weight": 0.5,
            "args": {"normalize_factor": 100000}
        },
        "carbon_emissions": {
            "weight": 0.3,
            "args": {"normalize_factor": 100}
        },
        "sla_penalty": {
            "weight": 0.2,
            "args": {"penalty_per_violation": 5.0}
        }
    }
)
```


---

## 🕒 SLA Modeling

GreenDCC includes built-in SLA (Service-Level Agreement) constraints to evaluate how well policies meet time-sensitive requirements.

Each task has a **deadline** computed as:


```python
SLA Deadline = task_start_time + SLA_FACTOR x task_duration
```


By default, `SLA_FACTOR = 1.2`, which means tasks are expected to finish **within 20% of their nominal runtime**.

This approach is inspired by the methodology used in:

> *Sustainable AIGC Workload Scheduling of Geo-Distributed Data Centers: A Multi-Agent Reinforcement Learning Approach*  
> [https://arxiv.org/abs/2304.07948](https://arxiv.org/abs/2304.07948)

In that paper, the authors simulate job slack times proportional to job duration — a structure also mirrored here.

### SLA Violation Penalty

You can include an `sla_penalty` reward to penalize missed deadlines:

```python
"sla_penalty": {
    "weight": 0.2,
    "args": {"penalty_per_violation": 5.0}
}
```
This allows policies to be evaluated based on both sustainability **and** reliability metrics.


---


## Why this matters

Not all users have the same priorities:

- **Cloud providers** might care about minimizing **energy price** and **resource efficiency**.
- **Sustainability-focused** deployments may want to reduce **carbon emissions** or **energy consumption**.
- Others may want to enforce strict **SLA guarantees**.

With our modular system, you can **define custom reward combinations** that align with your specific optimization objectives.


## Evaluation Modes

You can evaluate different controllers by plugging them into the DatacenterClusterManager. Available strategies:

- random
- round_robin
- most_available
- least_pending
- lowest_carbon
- lowest_price
- manual_rl (custom RL policy)

---

## ✅ Google Colab Evaluation

A ready-to-run Google Colab notebook is available for testing and evaluation:

👉 **[Run it here](https://colab.research.google.com/drive/1LLw313sG56l2I29E0Q9zh6KM0q5Z23WX?usp=sharing)**

The notebook supports:
- Uploading a trained agent checkpoint
- Running simulation for 7 days
- Comparing with a rule-based controller
- Visualizing energy, carbon, and resource usage

---
## TODO / Roadmap

- Add transmission cost and delay matrices
- GPU + Memory energy modeling
- More rule-based controllers for baseline comparison
- Modular reward builder
- Task delay & deferral logic
- Improve the Google colab notebook

---

## Citation / Credits

This project builds on work from:

- Alibaba Cluster Trace Dataset
- Electricity Maps Dataset
- Open-Meteo API
- GridStatus.io

---

## License

MIT License. Attribution to original dataset sources is required.

---

## Contributors

Feel free to open issues or PRs for improvements, bugs, or suggestions.
