.. _custom-reward-details:

customreward
============

Creating Custom Rewards
-----------------------

Users can implement novel reward functions tailored to specific research questions.

1. **Create File:**  
   Add a new Python file in the ``rewards/predefined/`` directory (e.g.,  
   ``rewards/predefined/my_custom_reward.py``).

2. **Inherit & Implement:**  
   Define a class inheriting from ``rewards.base_reward.BaseReward``.  
   Implement the ``__call__(self, cluster_info, current_tasks, current_time)``  
   method containing your custom logic. Optionally store the result in  
   ``self.last_reward``.

   .. code-block:: python

      # rewards/predefined/my_custom_reward.py
      from rewards.base_reward import BaseReward
      from rewards.registry_utils import register_reward
      import numpy as np

      @register_reward("my_custom")  # Choose a unique name
      class MyCustomReward(BaseReward):
          def __init__(self, custom_param=1.0):
              super().__init__()
              self.custom_param = custom_param

          def __call__(self, cluster_info, current_tasks, current_time):
              # Example: Penalize variance in CPU utilization across DCs
              cpu_utils = [
                  info["__common__"]["cpu_util_percent"]
                  for info in cluster_info["datacenter_infos"].values()
              ]
              util_variance = np.var(cpu_utils) if cpu_utils else 0
              reward = -util_variance * self.custom_param
              self.last_reward = reward
              return reward

3. **Register:**  
   Decorate your class with ``@register_reward("your_unique_name")``  
   (imported from ``rewards.registry_utils``).

4. **Import in Registry:**  
   Add an import for your new class at the top of  
   ``rewards/reward_registry.py`` so that it is discovered at runtime.

   .. code-block:: python

      # rewards/reward_registry.py
      # ... other imports ...
      from rewards.predefined.my_custom_reward import MyCustomReward  # Add this line

Once implemented and imported, your custom reward is available by name to  
``CompositeReward`` and ``get_reward_function``, and can be referenced in  
the ``components`` section of your ``reward_config.yaml``.
