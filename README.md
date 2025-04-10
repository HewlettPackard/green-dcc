
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
## 🧠 Action Space: How the Agent Makes Decisions

In this benchmark, at every decision step, the agent (**Global Scheduler**) is presented with a list of pending tasks. It must decide, for each task, what to do next.

The **action space** is defined as:

```python
action ∈ {0, 1, 2, ..., N}
```

Where:
- **N** is the total number of datacenters in the simulation (e.g., 5).
- The **action** is an integer that represents the decision for a given task.

### What Each Action Means

| Action Value | Meaning |
|--------------|---------|
| `0`          | **Defer the task**: Temporarily hold the task to be reconsidered in the next time step. This allows the agent to wait for better scheduling conditions (e.g., cheaper, greener, or less loaded datacenter). |
| `1` to `N`   | **Assign to datacenter `i`**: Send the task to the selected datacenter (e.g., `1 = DC1`, `2 = DC2`, ...). The task will enter that datacenter’s scheduling queue and execute when resources are available. |

### Why Deferring Matters

Deferring (action `0`) enables **temporal flexibility**. It gives the agent an option to wait for:
- **Lower carbon intensity**
- **Cheaper electricity prices**
- **Higher resource availability**

However, every task has a **deadline (SLA)**. If it waits too long, it will **violate the SLA** and may incur a penalty.

This flexible action space supports **more intelligent and sustainability-aware scheduling strategies**.

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

## Why this matters

Not all users have the same priorities:

- **Cloud providers** might care about minimizing **energy price** and **resource efficiency**.
- **Sustainability-focused** deployments may want to reduce **carbon emissions** or **energy consumption**.
- Others may want to enforce strict **SLA guarantees**.

With our modular system, you can **define custom reward combinations** that align with your specific optimization objectives.
This flexibility allows you to tailor the training process to your unique needs, whether you're focused on cost, carbon emissions, etc. or a combination of multiple factors.

---

## Installation
### Requirements
```bash
pip install -r requirements.txt
```
### Environment Setup
```bash
conda create -n green-dcc python=3.10
conda activate green-dcc
pip install -r requirements.txt
```

---
# 🏋️‍♂️ Training the RL Agent in GreenDCC

GreenDCC supports training Deep Reinforcement Learning agents using **Soft Actor-Critic (SAC)**. The training loop is fully configurable via YAML files and supports easy experimentation across different sustainability and efficiency goals.

---

## 📁 Configuration Overview

Training is driven by four modular config files:

| Config File | Purpose |
|-------------|---------|
| `sim_config.yaml` | Simulation time, strategy, task sources |
| `datacenters.yaml` | DCs with location, resource specs, energy model |
| `reward_config.yaml` | Reward weights for carbon, cost, SLA, etc. |
| `algorithm_config.yaml` | RL hyperparameters (batch size, learning rate, etc.) |

---

## 🚀 Start Training

Default:

```bash
python train_rl_agent.py
```

With custom paths:

```bash
python train_rl_agent.py \
    --sim-config configs/env/sim_config.yaml \
    --dc-config configs/env/datacenters.yaml \
    --reward-config configs/env/reward_config.yaml \
    --algo-config configs/env/algorithm_config.yaml \
    --tag my_experiment
```

Use `--checkpoint-path` to resume from a previous run.

---

## 🧠 RL Algorithm

The default training method is **Soft Actor-Critic (SAC)**, which features:

- Stable off-policy learning
- Entropy-based exploration
- Replay buffer optimization

The agent learns to **defer or route tasks** for better long-term trade-offs in carbon, cost, and load balancing.

---

## 📦 Checkpointing

Model checkpoints are saved in:

```
checkpoints/train_<timestamp>/
├── checkpoint_step_5000.pth
├── checkpoint_step_10000.pth
└── best_checkpoint.pth
```

Use them to resume training or for evaluation.

---

## 📈 Monitoring with TensorBoard

```bash
tensorboard --logdir runs/
```

Key logs include:

- Reward trends
- Q-loss and policy loss
- Entropy over time
- Reward components breakdown (carbon, price, SLA, etc.)

---

## 🔧 Customize Everything

Want to test a new reward? Just edit `reward_config.yaml`.

Want a different datacenter mix? Update `datacenters.yaml`.

Want faster updates or a longer warmup? Modify `algorithm_config.yaml`.

GreenDCC’s config-driven design makes it easy to explore new ideas.

---

## 📈 Tracking Training with TensorBoard

GreenDCC logs all major training metrics to **TensorBoard**, including:

- Total and per-step rewards
- Actor policy entropy
- Q-value estimates and loss
- Policy gradients and loss
- Reward component breakdowns (for composite rewards)

### 🏁 How to Launch

From the root directory:

```bash
tensorboard --logdir runs/
```

Then navigate to:

👉 http://localhost:6006

Each run logs to:

```
runs/train_<timestamp>/
```

You can compare multiple runs simultaneously for performance diagnostics or ablations.

---
## Customize Everything

Want to test a new reward? Just edit `reward_config.yaml`.

Want a different datacenter mix? Update `datacenters.yaml`.

Want faster updates or a longer warmup? Modify `algorithm_config.yaml`.

GreenDCC’s config-driven design makes it easy to explore new ideas.

---

## Evaluation

### Evaluation using Rule-based Controllers or RL Agents

You can evaluate different controllers by plugging them into the DatacenterClusterManager. Available strategies:

- random
- round_robin
- most_available
- least_pending
- lowest_carbon
- lowest_price
- manual_rl (custom RL policy)

To run the evaluation, take a look to:

```bash
python eval_agent_notebook.py
```
This will run a simulation for 7 days and compare the performance of the selected controller against a rule-based controller.
You need to specify the controller you want to evaluate in the `controller` variable.
Also you need to specify the checkpoint if using a RL agent.

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
