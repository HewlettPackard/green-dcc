===============================
Benchmarking Algorithms
===============================


The |F| environment supports benchmarking various Multi Agent / Hierarchical control algorithms to evaluate their effectiveness in optimizing workload distribution and minimizing the carbon footprint of data center clusters. This section provides instructions on how to run benchmarks using different algorithms and configurations.


Supported Algorithms
--------------------------

While |F| is compatible with a wide range of algorithms provided by Ray RLlib, our experiments have primarily tested and validated the following algorithms:

  - **Advantage Actor-Critic (A2C)**
  - **Adaptive Proximal Policy Optimization (APPO)**
  - **Proximal Policy Optimization (PPO)**

These algorithms have been successfully trained and evaluated within the |F| environment, demonstrating their performance in terms of energy consumption, carbon footprint, and other relevant metrics.

Other algorithms listed in the Ray RLlib `documentation <https://docs.ray.io/en/releases-2.4.0/rllib/rllib-algorithms.html>`_ should also be compatible with |F|, but additional work may be required to adapt the environment to the expected input and output shapes of each method as implemented in RLlib. For more details on these algorithms and how to adapt them for |F|, refer to the Ray RLlib `documentation <https://docs.ray.io/en/releases-2.4.0/rllib/rllib-algorithms.html>`_.
