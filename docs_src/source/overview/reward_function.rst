.. _reward-functions:

Reward Functions
================

Sustain-Cluster’s reward framework defines how the Top-Level Agent is incentivized to make spatio-temporal scheduling decisions. All reward classes inherit from ``rewards.base_reward.BaseReward`` and are registered via the ``@register_reward(name)`` decorator. At each simulation step, the agent computes a scalar reward by calling the selected reward function with the current cluster state, the list of tasks under consideration, and the current timestamp.


Base Class and Registry
-----------------------

.. py:class:: BaseReward(**kwargs**)

   Abstract base class for all rewards. Subclasses must implement:

   - ``__call__(cluster_info: dict, current_tasks: list, current_time: Any) -> float``
   - ``get_last_value() -> float``

   Common behavior:

   - Stores the last computed reward in ``self.last_reward``.
   - Supports arbitrary constructor arguments via ``**kwargs``.

.. note::

   To expose a custom reward, subclass ``BaseReward`` and annotate with
   ``@register_reward("your_reward_name")``. The class will then be discoverable
   via ``get_reward_function("your_reward_name", **args)``.


Built-in Reward Classes
-----------------------

Carbon Emissions Reward
~~~~~~~~~~~~~~~~~~~~~~~

.. py:class:: CarbonEmissionsReward(normalize_factor: float = 100.0)

   **Parameters**

   - **normalize_factor** (float) – Divisor to scale total CO₂ (kg) into reward units.

   **Description**

   Penalizes total carbon emissions across the cluster.

   **Computation**

   1. Sum ``dc_info["__common__"]["carbon_emissions_kg"]`` for every datacenter.  
   2. Compute ``reward = - total_emissions / normalize_factor``.  
   3. Store value in ``self.last_reward``.

   .. math::

      E_{\mathrm{tot}} = \sum_{d \in D} e_{d},\quad
      R = -\frac{E_{\mathrm{tot}}}{\mathrm{normalize\_factor}}

   **Variables**

   - :math:`D` – Set of all datacenters.  
   - :math:`e_{d}` – Carbon emissions (kg) of datacenter :math:`d`.  
   - :math:`E_{\mathrm{tot}}` – Total carbon emissions across :math:`D`.  
   - :math:`\mathrm{normalize\_factor}` – Constructor parameter.  
   - :math:`R` – Resulting reward.


Energy Consumption Reward
~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:class:: EnergyConsumptionReward(normalize_factor: float = 1000.0)

   **Parameters**

   - **normalize_factor** (float) – Divisor to scale total kWh into reward units.

   **Description**

   Penalizes total energy usage across all datacenters.

   **Computation**

   1. Sum ``dc_info["__common__"]["energy_consumption_kwh"]`` for every datacenter.  
   2. Compute ``reward = - total_energy / normalize_factor``.  
   3. Store in ``self.last_reward``.

   .. math::

      E_{\mathrm{tot}} = \sum_{d \in D} \mathrm{energy\_consumption\_kWh}_{d},\quad
      R = -\frac{E_{\mathrm{tot}}}{\mathrm{normalize\_factor}}

   **Variables**

   - :math:`D` – Set of all datacenters.  
   - :math:`\mathrm{energy\_consumption\_kWh}_{d}` – Energy consumed (kWh) by datacenter :math:`d`.  
   - :math:`E_{\mathrm{tot}}` – Total energy consumption across :math:`D`.  
   - :math:`R` – Resulting reward.


Energy Price Reward
~~~~~~~~~~~~~~~~~~~

.. py:class:: EnergyPriceReward(normalize_factor: float = 100000)

   **Parameters**

   - **normalize_factor** (float) – Divisor to scale USD cost into reward units.

   **Description**

   Penalizes monetary cost of energy consumed by scheduled tasks, using real-time prices.

   **Computation**

   1. For each task in ``current_tasks``:

      - Retrieve ``price = dest_dc.price_manager.get_current_price()``.  
      - Compute ``task_energy = task.cores_req * task.duration`` (kWh).  
      - Compute ``task_cost = task_energy * price``.  

   2. Sum all ``task_cost`` values.  
   3. Compute ``reward = - total_task_cost / normalize_factor``.  
   4. Store in ``self.last_reward``.

   .. math::

      C_{\mathrm{tot}} = \sum_{t \in T} p_{t}\,c_{t}\,\tau_{t},\quad
      R = -\frac{C_{\mathrm{tot}}}{\mathrm{normalize\_factor}}

   **Variables**

   - :math:`T` – Set of tasks in ``current_tasks``.  
   - :math:`p_{t}` – Price (USD/kWh) returned by ``dest_dc.price_manager.get_current_price()`` for task :math:`t`.  
   - :math:`c_{t}` – ``task.cores_req`` (number of cores) for task :math:`t`.  
   - :math:`\tau_{t}` – ``task.duration`` (hours) for task :math:`t`.  
   - :math:`C_{\mathrm{tot}}` – Total energy cost (USD).  
   - :math:`R` – Resulting reward.


SLA Penalty Reward
~~~~~~~~~~~~~~~~~~

.. py:class:: SLAPenaltyReward(penalty_per_violation: float = 10.0)

   **Parameters**

   - **penalty_per_violation** (float) – Penalty per SLA breach.

   **Description**

   Penalizes missed service-level agreements across the cluster.

   **Computation**

   1. Count violations across all datacenters:  
      ``violations = sum(dc_info["__common__"]["__sla__"]["violated"] for dc_info in cluster_info["datacenter_infos"].values())``  
   2. Compute ``reward = - penalty_per_violation * violations``.  
   3. Store in ``self.last_reward``.

   .. math::

      V = \sum_{d \in D} v_{d},\quad
      R = -\,\mathrm{penalty\_per\_violation}\;\times V

   **Variables**

   - :math:`D` – Set of all datacenters.  
   - :math:`v_{d}` – Number of SLA violations in datacenter :math:`d`.  
   - :math:`V` – Total SLA violations across :math:`D`.  
   - :math:`\mathrm{penalty\_per\_violation}` – Constructor parameter.  
   - :math:`R` – Resulting reward.


Transmission Cost Reward
~~~~~~~~~~~~~~~~~~~~~~~~

.. py:class:: TransmissionCostReward(normalize_factor: float = 100.0)

   **Parameters**

   - **normalize_factor** (float) – Divisor to scale USD transmission cost.

   **Description**

   Penalizes cumulative inter-datacenter bandwidth costs.

   **Computation**

   1. Read ``cost = cluster_info["transmission_cost_total_usd"]``.  
   2. Compute ``reward = - cost / normalize_factor``.  
   3. Store in ``self.last_reward``.

   .. math::

      C = \mathrm{transmission\_cost\_total\_usd},\quad
      R = -\frac{C}{\mathrm{normalize\_factor}}

   **Variables**

   - :math:`C` – Total inter-datacenter transmission cost (USD).  
   - :math:`R` – Resulting reward.


Transmission Emissions Reward
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:class:: TransmissionEmissionsReward(normalize_factor: float = 1.0)

   **Parameters**

   - **normalize_factor** (float) – Divisor to scale kg CO₂ from transmission.

   **Description**

   Penalizes carbon emissions incurred by data transfer between datacenters.

   **Computation**

   1. Read ``emissions_kg = cluster_info["transmission_emissions_total_kg"]``.  
   2. Compute ``reward = - emissions_kg / normalize_factor``.  
   3. Store in ``self.last_reward``.

   .. math::

      E_{\mathrm{tr}} = \mathrm{transmission\_emissions\_total_kg},\quad
      R = -\frac{E_{\mathrm{tr}}}{\mathrm{normalize\_factor}}

   **Variables**

   - :math:`E_{\mathrm{tr}}` – Total transmission emissions (kg CO₂).  
   - :math:`R` – Resulting reward.


Efficiency Reward
~~~~~~~~~~~~~~~~~

.. py:class:: EfficiencyReward(normalize_factor: float = 1000.0)

   **Parameters**

   - **normalize_factor** (float) – Divisor to scale energy per task.

   **Description**

   Encourages high energy efficiency per scheduled task.

   **Computation**

   1. Sum ``total_energy`` across datacenters.  
   2. Read ``total_tasks = cluster_info.get("scheduled_tasks", 0)``.  
   3. If ``total_tasks == 0``, return 0.  
   4. Compute ``reward = - (total_energy / total_tasks)``.  
   5. Store in ``self.last_reward``.

   .. math::

      E_{\mathrm{tot}} = \sum_{d \in D} \mathrm{energy\_consumption\_kWh}_{d},\quad
      N = \mathrm{total\_tasks},\quad
      R = -\frac{E_{\mathrm{tot}}}{N}

   **Variables**

   - :math:`N` – Number of scheduled tasks.  
   - :math:`E_{\mathrm{tot}}` – Total energy consumption (kWh).  
   - :math:`R` – Resulting reward.


Composite Reward
----------------

.. py:class:: CompositeReward(components: dict, normalize: bool = True, epsilon: float = 1e-8)

   **Parameters**

   - **components** (dict) – Mapping from reward name to a dict with keys:  
     - **weight** (float)  
     - **args** (constructor kwargs)  
   - **normalize** (bool) – If True, z-score each component.  
   - **epsilon** (float) – Small constant to avoid division by zero.

   **Description**

   Combines multiple reward signals into a single scalar via a weighted sum.

   **Internal State**

   - ``running_stats`` – Per-component running mean, variance, and count.  
   - ``last_values`` – Last raw values before normalization.

   **Computation**

   1. For each `(name, weight, fn)` in `components`, call:  
      ``raw = fn(cluster_info, current_tasks, current_time)``.  
   2. If `normalize` is True, update running stats for `name` and compute  
      ``component_value = (raw - mean) / (std + epsilon)``, otherwise set  
      ``component_value = raw``.  
   3. Add ``weight * component_value`` to `total`.  
   4. Set ``self.last_reward = total`` and return `total`.

   .. math::

      \hat{v}_{i} =
      \begin{cases}
        \dfrac{raw_{i} - \mu_{i}}{\sigma_{i} + \epsilon}, & \text{if normalize} \\
        raw_{i}, & \text{otherwise}
      \end{cases},\quad
      R = \sum_{i} w_{i}\,\hat{v}_{i}

   **Variables**

   - :math:`raw_{i}` – Raw value of component :math:`i`.  
   - :math:`\mu_{i}, \sigma_{i}` – Running mean and standard deviation of component :math:`i`.  
   - :math:`\epsilon` – Small constant to avoid division by zero.  
   - :math:`\hat{v}_{i}` – (Possibly normalized) component value.  
   - :math:`w_{i}` – Weight for component :math:`i`.  
   - :math:`R` – Resulting composite reward.


Registry and Invocation
-----------------------

.. code-block:: python

   from rewards.registry_utils import get_reward_function

   reward_fn = get_reward_function("energy_price", normalize_factor=50000)
   value   = reward_fn(cluster_info, current_tasks, current_time)
   raw_val = reward_fn.get_last_value()
