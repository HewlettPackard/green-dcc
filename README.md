
# Multi-Data Center Geographical Scheduling Benchmark

A high-fidelity benchmark for multi-objective scheduling across globally distributed data centers, optimized for energy efficiency, carbon footprint, transfer cost, and operational sustainability.

> Goal: Given real-world AI workloads originating from multiple datacenters, assign each task to the best destination datacenter, balancing multiple sustainability and operational trade-offs.

---

## Features

- Centralized global scheduler with distributed task generation
- Realistic simulation using real-world datasets (Alibaba, ElectricityMaps, Open-Meteo)
- Transfer-aware task routing with bandwidth cost and latency modeling
- Dynamic sustainability metrics:
  - Carbon intensity
  - Energy prices
  - Weather (temperature, cooling)
  - Water use (via proxies)
- Modular design with support for:
  - Rule-based controllers
  - Reinforcement learning (SAC-based implemented)
- Easy to extend and evaluate new scheduling strategies

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

## Objectives to Optimize

- Energy efficiency
- Carbon footprint
- Water usage (proxy via temperature/cooling)
- Bandwidth and latency
- Economic cost (price per kWh, transfer fees)

---

## Datasets Used

| Type               | Source                                                                 |
|--------------------|------------------------------------------------------------------------|
| AI Workloads       | Alibaba Cluster Trace 2020                                             |
| Weather Data       | Open-Meteo                                                             |
| Carbon Intensity   | Electricity Maps                                                       |
| Energy Prices      | Mixed sources: Electricity Maps, GridStatus, Open APIs (per country)   |

---

## Dataset Format

The cleaned dataset is saved as a Pandas .pkl file with the following structure:

interval_15m | tasks_matrix  
-------------|----------------------------------------  
2020-03-01   | [[job1, tstart, tend, cpu, gpu, mem, bw], ...]

Task fields:
- cpu_usage, gpu_util, mem, bandwidth_gb
- origin_dc_id assigned using hybrid population × local-time logic

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

# ⚙️ Modular Reward System

We provide a **flexible reward function framework** so users can optimize scheduling policies for **their own sustainability goals** — whether that’s minimizing energy cost, reducing carbon footprint, improving SLA, or a combination.

## Why this matters

Not all users have the same priorities:

- **Cloud providers** might care about minimizing **energy price** and **resource efficiency**.
- **Sustainability-focused** deployments may want to reduce **carbon emissions** or **energy consumption**.
- Others may want to enforce strict **SLA guarantees**.

With our modular system, you can **define custom reward combinations** that align with your specific optimization objectives.

---

## ✅ Built-in Reward Functions

The following reward components are already available:

| Reward Name        | Description                                      | Params                     |
|--------------------|--------------------------------------------------|----------------------------|
| `energy_price`     | Penalizes total energy cost of tasks             | `normalize_factor`         |
| `carbon_emissions` | Penalizes total kgCO₂ emissions                  | `normalize_factor`         |
| `energy_consumption` | Penalizes total energy used (in kWh)          | `normalize_factor`         |
| `efficiency`       | Penalizes energy per scheduled task              | _None_                     |
| `sla_penalty`      | Penalizes number of SLA violations               | `penalty_per_violation`    |
| `composite`        | Combines multiple reward components              | See below                  |

---

## 🔧 Example: Composite Reward

To combine multiple goals (e.g. cost, carbon, SLA), use `CompositeReward`:

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

Then pass it to the environment:

```python
env = TaskSchedulingEnv(
    cluster_manager=cluster_manager,
    start_time=start_time,
    end_time=end_time,
    reward_fn=reward_fn
)
```

---

## 📁 Reward Folder Structure

All reward logic lives under:

```
rewards/
├── base_reward.py               # Reward interface
├── reward_registry.py           # Auto-registers reward classes
├── registry_utils.py            # Registry implementation
├── predefined/
│   ├── energy_price_reward.py
│   ├── carbon_emissions_reward.py
│   ├── energy_consumption_reward.py
│   ├── sla_penalty_reward.py
│   ├── efficiency_reward.py
│   └── composite_reward.py
```

Each reward is automatically registered using decorators like:

```python
@register_reward("energy_price")
class EnergyPriceReward(BaseReward):
    ...
```

This allows easy instantiation anywhere via:

```python
from rewards.reward_registry import get_reward_function

reward_fn = get_reward_function("energy_price", normalize_factor=100000)
```

---

## ✍️ Custom Rewards

You can add your own reward in `rewards/predefined/`:

```python
@register_reward("my_custom_reward")
class MyReward(BaseReward):
    def __call__(self, cluster_info, current_tasks, current_time):
        ...
```

It becomes available automatically via `get_reward_function("my_custom_reward")`.

---

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
