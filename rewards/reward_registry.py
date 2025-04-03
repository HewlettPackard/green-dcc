# rewards/reward_registry.py

from rewards.registry_utils import get_reward_function, reward_registry

# Import all reward classes
from rewards.predefined.energy_price_reward import EnergyPriceReward
from rewards.predefined.sla_penalty_reward import SLAPenaltyReward
from rewards.predefined.efficiency_reward import EfficiencyReward
from rewards.predefined.composite_reward import CompositeReward
from rewards.predefined.carbon_emissions_reward import CarbonEmissionsReward
from rewards.predefined.energy_consumption_reward import EnergyConsumptionReward
