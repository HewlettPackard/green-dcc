# Reward System in SustainCluster

SustainCluster enables systematic comparison of scheduling policies across **environmental**, **economic**, and **operational** trade-offs.

This reward system is a key component of the benchmark, enabling:
- reproducible comparisons
- configurable multi-objective settings
- fairness across different optimization goals

---

## Reward Definitions

Reward functions are implemented in `rewards/predefined/` and inherit from the `BaseReward` class. Each reward implements a `__call__` method that returns a reward value based on the environment state.

### Available Rewards

| Reward Name         | File                                 | Description                                   |
|---------------------|--------------------------------------|-----------------------------------------------|
| `energy_price`      | energy_price_reward.py               | Penalizes high-cost energy usage              |
| `carbon_emissions`  | carbon_emissions_reward.py           | Penalizes CO₂ emissions across DCs            |
| `energy_consumption`| energy_consumption_reward.py         | Penalizes high total energy consumption       |
| `efficiency`        | efficiency_reward.py                 | Rewards more tasks per unit of energy         |
| `sla_penalty`       | sla_penalty_reward.py                | Penalizes SLA violations                      |
| `transmission_cost` | transmission_cost_reward.py          | Penalizes high inter-DC transmission costs    |
| `composite`         | composite_reward.py                  | Combines multiple rewards with weights        |

---

## 📁 Reward Folder Structure

```
rewards/
├── base_reward.py               # Abstract base class for all reward functions
├── reward_registry.py           # Registry system for dynamically loading rewards
├── registry_utils.py            # Decorators + fetch methods
└── predefined/                  # Predefined reward functions
    ├── carbon_emissions_reward.py
    ├── energy_price_reward.py
    ├── energy_consumption_reward.py
    ├── efficiency_reward.py
    ├── sla_penalty_reward.py
    ├── transmission_cost_reward.py
    └── composite_reward.py      # Combine multiple sub-rewards
```

---

## Base Class (All rewards inherit from)

```python
class BaseReward:
    def __call__(self, cluster_info, current_tasks, current_time):
        raise NotImplementedError
```

---

## Using a Reward Function

To use a reward in training:

```python
from rewards.predefined.energy_price_reward import EnergyPriceReward

reward_fn = EnergyPriceReward(normalize_factor=100000)
```

For a composite reward:

```python
from rewards.predefined.composite_reward import CompositeReward

reward_fn = CompositeReward(
    components={
        "energy_price": {"weight": 0.5, "args": {"normalize_factor": 100000}},
        "carbon_emissions": {"weight": 0.3, "args": {"normalize_factor": 10}},
        "transmission_cost": {"weight": 0.2, "args": {"normalize_factor": 1}}
    },
    normalize=False
)
```

Then pass this into your environment:

```python
env = TaskSchedulingEnv(
    cluster_manager=cluster_manager,
    start_time=start_time,
    end_time=end_time,
    reward_fn=reward_fn
)
```

---

## Registry and Auto-Loading

Each reward is registered using the `@register_reward("name")` decorator and added to the `reward_registry`. This allows dynamic loading of rewards by name:

```python
from rewards.registry_utils import get_reward_function

fn = get_reward_function("energy_price", normalize_factor=100000)
```

This is used internally by the `CompositeReward`.

---

## High level Reward Diagram

```
                ┌──────────────┐
                │ Cluster Info │
                └─────┬────────┘
                      │
            ┌─────────▼───────────┐
            │     Reward Engine   │
            └──────────┬──────────┘
                       │
        ┌──────────────┼─────────────┐
        │              │             │
   ┌────▼─────┐  ┌─────▼────┐ ┌──────▼──────┐
   │ energy   │  │ carbon   │ │ transmission│   ...
   │  price   │  │ emissions│ │    cost     │ 
   └──────────┘  └──────────┘ └─────────────┘
         │             │             │
    ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
    | weights |   | weights |   | weights |
    └─────────┘   └─────────┘   └─────────┘
         │             │             │
          ────► Combined Reward ◄────
                       ↓
              Backprop / Logging
```

---

## Custom Rewards

You can add new reward functions by:

1. Creating a file in `rewards/predefined/`
2. Inheriting from `BaseReward`
3. Registering it using `@register_reward("your_name")`
4. **IMPORTANT:** Also import your reward class in `rewards/reward_registry.py`
   to make sure the `@register_reward(...)` decorator gets executed at runtime.

Example:

```python
# rewards/predefined/my_custom_reward.py

from rewards.base_reward import BaseReward
from rewards.registry_utils import register_reward

@register_reward("my_custom")
class MyCustomReward(BaseReward):
    def __call__(self, cluster_info, current_tasks, current_time):
        reward = -some_metric
        self.last_reward = reward
        return reward
```

Then register it in `rewards/reward_registry.py`:

```python
# rewards/reward_registry.py

from rewards.predefined.my_custom_reward import MyCustomReward
```

---

## Reward Logging Utilities

Most reward classes support internal tracking of the last computed reward value. This feature is useful for logging, debugging, or visualizing individual reward signals (e.g., with TensorBoard).

### `self.last_reward`

Each reward class can store the latest reward in an internal variable `self.last_reward` when the reward is calculated in the `__call__` method. This allows the environment to retrieve the last computed value without recalculating it.

Example:

```python
def __call__(self, cluster_info, current_tasks, current_time):
    reward = -total_energy / self.normalize_factor
    self.last_reward = reward
    return reward
```

### `get_last_value()`

This method is defined in the reward class to return the value stored in `self.last_reward`.

Example:

```python
def get_last_value(self):
    return self.last_reward
```

### How it's used

In the environment (`TaskSchedulingEnv.step()`), we log rewards using `get_last_value()` when available. This allows per-step logging in TensorBoard.

Example:

```python
if self.writer:
    if hasattr(self.reward_fn, "get_last_value"):
        self.writer.add_scalar(f"Reward/{str(self.reward_fn)}", self.reward_fn.get_last_value(), self.global_step)
```

This enables easy visualization of the reward signal across time for each training run.

## Key Points

- Enables **multi-objective optimization** (modularity)
- Plug-in design for **custom, domain-specific reward terms**
- **Normalized, weighted reward combination**
- Clean separation between **rewards and evaluation metrics**
- Perfect for **ablation studies** and **reproducibility**

> Reward modularity is central to building credible and flexible benchmarks.

---

Feel free to explore `rewards/predefined/` and extend with your domain-specific logic.