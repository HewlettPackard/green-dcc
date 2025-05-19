.. _environments:

Sustain-Cluster Environment
=====================

State Observation
-----------------

At each timestep :math:`t`, the environment returns a detailed, variable-length observation :math:`s_t`. This observation is structured as a list of :math:`k_t` task-feature vectors, one per pending task.

Per-Task Vector  
By default (in ``_get_obs`` of ``TaskSchedulingEnv``), each pending task :math:`i` is represented by a concatenated feature vector of length :math:`4 + 5 + 5N` where:

- **Global Time Features** (4 features):  
  Sine/cosine encoding of the day of year and hour of day.

- **Task-Specific Features** (5 features):  
  Origin DC ID, CPU-core requirement, GPU requirement, estimated duration, and time remaining until SLA deadline.

- **Per-Datacenter Features** (5 × N features):  
  For each of the :math:`N` datacenters: available CPU %, available GPU %, available memory %, current carbon intensity (kg CO₂/kWh), and current electricity price (USD/kWh).

Because the number of pending tasks :math:`k_t` can change between timesteps (i.e. :math:`k_t ≠ k_{t+1}`), the overall shape of :math:`s_t` varies. For example, :math:`s_t` might be a list of 10 vectors (10 tasks), while :math:`s_{t+1}` might be only 5 vectors (5 tasks).

Handling Variability  
- **Off-policy SAC agents** use a ``FastReplayBuffer`` that pads each list of observations to a fixed ``max_tasks`` length and applies masking during batch updates.  
- **On-policy agents** (e.g. A2C) can process the variable-length list sequentially during rollouts, aggregating per-task values into a single state value.

Customization  
Users may override ``_get_obs`` to include additional information from ``self.cluster_manager.datacenters`` (e.g. pending queue lengths, detailed thermal state, forecasted CI) or ``self.current_tasks`` to craft bespoke state representations tailored to their agent architecture, scheduling strategy, or reward function.

Action Space
------------

At each timestep :math:`t`, the agent receives the list of :math:`k_t` pending tasks and must output one discrete action per task::

  a_i ∈ {0, 1, …, N}

- **0**: defer the :math:`i`-th task (it remains pending and is reconsidered in the next 15-minute step).  
- **j** (where :math:`1 ≤ j ≤ N`): assign the :math:`i`-th task to datacenter :math:`j` (incurring any transmission cost or delay if :math:`j` differs from the task’s origin).

Since :math:`k_t` varies over time, the action requirement per timestep is also variable-length. See **State Observation** for how existing RL examples accommodate this in both off-policy and on-policy settings.

Reward Signal
-------------

After all :math:`k_t` actions for timestep :math:`t` are applied and the simulator advances, a single global scalar reward :math:`r_t` is returned. This reward is computed by a configurable **RewardFunction** (see :ref:`reward-functions`), which aggregates performance and sustainability metrics according to user-defined weights and objectives, for example:

- Minimizing operational cost  
- Minimizing carbon footprint  
- Minimizing total energy consumption  
- Minimizing SLA violations  

Users may extend or replace the default reward function to reflect custom operational goals and trade-offs.
