
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

Global Scheduler  
      ↑  
  [All Tasks]  
      ↑  
+----------+   +----------+   +----------+  
|   DC1    |   |   DC2    |   |   DC3    |   ...

Each DC: local environment with
- Resource tracking
- Energy & carbon simulation
- Scheduling queue

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

## TODO / Roadmap

- Add transmission cost and delay matrices
- GPU + Memory energy modeling
- More rule-based controllers for baseline comparison
- Modular reward builder
- Task delay & deferral logic

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
