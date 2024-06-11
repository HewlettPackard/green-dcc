# Green-DCC: Benchmarking Dynamic Workload Distribution Techniques for Sustainable Data Center Cluster

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Benchmarking](#benchmarking)
5. [Experimental Details](#experimental-details)
6. [Selected Locations](#selected-locations)
7. [Contributing](#contributing)
8. [Contact](#contact)
9. [License](#license)

## Introduction

Green-DCC is a benchmark environment designed to evaluate dynamic workload distribution techniques for sustainable Data Center Clusters (DCC). It aims to reduce the environmental impact of cloud computing by distributing workloads within a DCC that spans multiple geographical locations. The benchmark environment supports the evaluation of various control algorithms, including reinforcement learning-based approaches.

Key features of Green-DCC include:

- Dynamic time-shifting of workloads within data centers and geographic shifting between data centers in a cluster.
- Incorporation of non-uniform computing resources, cooling capabilities, auxiliary power resources, and varying external weather and carbon intensity conditions.
- A dynamic bandwidth cost model that accounts for the geographical characteristics and amount of data transferred.
- Realistic workload execution delays to reflect changes in data center capacity and demand.
- Support for benchmarking multiple heuristic and hierarchical reinforcement learning-based approaches.
- Customizability to address specific needs of cloud providers or enterprise data center clusters.

Green-DCC provides a complex, interdependent, and realistic benchmarking environment that is well-suited for evaluating hierarchical reinforcement learning algorithms applied to data center control. The ultimate goal is to optimize workload distribution to minimize the carbon footprint, energy usage, and energy cost, while considering various operational constraints and environmental factors.

Detailed documentation is available [here](https://hewlettpackard.github.io/green-dcc).

## Installation

To get started with Green-DCC, follow the steps below to set up your environment and install the necessary dependencies.

### Prerequisites

- Python 3.10+
- ray 2.4.0 (installed when installing the `requirements.txt` file)
- Git
- Conda (for creating virtual environments)

### Setting Up the Environment

1. **Clone the repository**

    First, clone the Green-DCC repository from GitHub:

    ```bash
    git clone https://github.com/HewlettPackard/green-dcc.git
    cd green-dcc
    ```

2. **Create a new Conda environment**

    Create a new Conda environment with Python 3.10:

    ```bash
    conda create --name greendcc python=3.10
    ```

3. **Activate the environment**

    Activate the newly created environment:

    ```bash
    conda activate greendcc
    ```

4. **Install dependencies**

    Install the required dependencies using `pip`:

    ```bash
    pip install -r requirements.txt
    ```


## Usage

This section provides instructions on how to run simulations, configure the environment, and use the Green-DCC benchmark.

### Running Simulations

1. **Navigate to the Green-DCC directory**

    Ensure you are in the `green-dcc` directory:

    ```bash
    cd green-dcc
    ```
    
2. **Run a simulation**

    To run a basic simulation, use the following command:

    ```bash
    python train_truly_hierarchical.py
    ```

    This will start a simulation with the default configuration. The results will be saved in `results/` output directory.

## Benchmarking

The Green-DCC environment supports benchmarking various Multi Agent / Hierarchical control algorithms to evaluate their effectiveness in optimizing workload distribution and minimizing the carbon footprint of data center clusters. This section provides instructions on how to run benchmarks using different algorithms and configurations.

### Tested Algorithms

While Green-DCC is compatible with a wide range of algorithms provided by Ray RLlib, our experiments have primarily tested and validated the following algorithms:

- **Advantage Actor-Critic (A2C)**
- **Adaptive Proximal Policy Optimization (APPO)**
- **Proximal Policy Optimization (PPO)**

These algorithms have been successfully trained and evaluated within the Green-DCC environment, demonstrating their performance in terms of energy consumption, carbon footprint, and other relevant metrics.

Other algorithms listed on the [Ray RLlib documentation](https://docs.ray.io/en/releases-2.4.0/rllib/rllib-algorithms.html) should also be compatible with Green-DCC, but additional work may be required to adapt the environment to the expected input and output shapes of each method as implemented in RLlib. For more details on these algorithms and how to adapt them for Green-DCC, refer to the [Ray RLlib documentation](https://docs.ray.io/en/releases-2.4.0/rllib/rllib-algorithms.html).



### Running Benchmarks

1. **Navigate to the Green-DCC directory**

    Ensure you are in the `green-dcc` directory:

    ```bash
    cd green-dcc
    ```

2. **Configure the benchmark**

    Edit the configuration files as needed to set up your desired benchmark parameters.

3. **Train and evaluate algorithms**

    To train and evaluate an RL algorithm using Ray, use the appropriate training script. For example, to train the PPO algorithm, run:

    ```bash
    python train_truly_hierarchical.py
    ```

    The provided training script (`train_truly_hierarchical.py`) uses Ray for distributed training. Here's a brief overview of the script for PPO:

    ```python
    import os
    import ray
    from ray import air, tune
    from ray.rllib.algorithms.ppo import PPO, PPOConfig
    from gymnasium.spaces import Discrete, Box
    from ray.rllib.algorithms.ppo import PPOConfig
    
    from envs.truly_heirarchical_env import TrulyHeirarchicalDCRL
    from envs.heirarchical_env import HeirarchicalDCRL, DEFAULT_CONFIG
    from create_trainable import create_wrapped_trainable
    
    NUM_WORKERS = 1
    NAME = "test"
    RESULTS_DIR = './results/'
    
    # Dummy env to get obs and action space
    hdcrl_env = HeirarchicalDCRL()
    
    CONFIG = (
            PPOConfig()
            .environment(
                env=TrulyHeirarchicalDCRL,
                env_config=DEFAULT_CONFIG
            )
            .framework("torch")
            .rollouts(
                num_rollout_workers=NUM_WORKERS,
                rollout_fragment_length=2,
                )
            .training(
                gamma=0.99,
                lr=1e-5,
                kl_coeff=0.2,
                clip_param=0.1,
                entropy_coeff=0.0,
                use_gae=True,
                train_batch_size=4096,
                num_sgd_iter=10,
                model={'fcnet_hiddens': [64, 64]}, 
                shuffle_sequences=True
            )
            .multi_agent(
            policies={
                "high_level_policy": (
                    None,
                    hdcrl_env.observation_space,
                    hdcrl_env.action_space,
                    PPOConfig()
                ),
                "DC1_ls_policy": (
                    None,
                    Box(-1.0, 1.0, (14,)),
                    Discrete(3),
                    PPOConfig()
                ),
                "DC2_ls_policy": (
                    None,
                    Box(-1.0, 1.0, (14,)),
                    Discrete(3),
                    PPOConfig()
                ),
                "DC3_ls_policy": (
                    None,
                    Box(-1.0, 1.0, (14,)),
                    Discrete(3),
                    PPOConfig()
                ),
            },
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: agent_id,
            )
            .resources(num_gpus=0)
            .debugging(seed=0)
        )


    if __name__ == "__main__":
        os.environ["RAY_DEDUP_LOGS"] = "0"
        # ray.init(local_mode=True, ignore_reinit_error=True)
        ray.init(ignore_reinit_error=True)
        
        tune.Tuner(
            create_wrapped_trainable(PPO),
            param_space=CONFIG.to_dict(),
            run_config=air.RunConfig(
                stop={"timesteps_total": 100_000_000},
                verbose=0,
                local_dir=RESULTS_DIR,
                # storage_path=RESULTS_DIR,
                name=NAME,
                checkpoint_config=ray.air.CheckpointConfig(
                    checkpoint_frequency=5,
                    num_to_keep=5,
                    checkpoint_score_attribute="episode_reward_mean",
                    checkpoint_score_order="max"
                ),
            )
    ).fit()  
    ```

    This example assumes a DCC with three data centers. To use a different algorithm, such as SAC, you need to replace the `PPOConfig` with `SACConfig` (or the appropriate config class for the algorithm) and adjust the hyperparameters accordingly. For example:

    ```python
    from ray.rllib.algorithms.a2c import A2C, A2CConfig

    CONFIG = (
            A2CConfig()
            .environment(
                env=TrulyHeirarchicalMSDCRL,
                env_config=DEFAULT_CONFIG
            )
            .framework("torch")
            .rollouts(
                num_rollout_workers=NUM_WORKERS,
                rollout_fragment_length=2,
                )
            .training(
                gamma=0.99,
                lr=1e-5,
                kl_coeff=0.2,
                clip_param=0.1,
                entropy_coeff=0.0,
                use_gae=True,
                train_batch_size=4096,
                num_sgd_iter=10,
                model={'fcnet_hiddens': [64, 64]}, 
                #shuffle_sequences=True
            )
            .multi_agent(
            policies={
                "high_level_policy": (
                    None,
                    hdcrl_env.observation_space,
                    hdcrl_env.action_space,
                    A2CConfig()
                ),
                "DC1_ls_policy": (
                    None,
                    Box(-1.0, 1.0, (14,)),
                    Discrete(3),
                    A2CConfig()
                ),
                "DC2_ls_policy": (
                    None,
                    Box(-1.0, 1.0, (14,)),
                    Discrete(3),
                    A2CConfig()
                ),
                "DC3_ls_policy": (
                    None,
                    Box(-1.0, 1.0, (14,)),
                    Discrete(3),
                    A2CConfig()
                ),
            },
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: agent_id,
            )
            .resources(num_gpus=0)
            .debugging(seed=1)
        )


    if __name__ == "__main__":
        os.environ["RAY_DEDUP_LOGS"] = "0"
        # ray.init(local_mode=True, ignore_reinit_error=True)
        ray.init(ignore_reinit_error=True)
        
        tune.Tuner(
            create_wrapped_trainable(A2C),
            param_space=CONFIG.to_dict(),
            run_config=air.RunConfig(
                stop={"timesteps_total": 100_000_000},
                verbose=0,
                local_dir=RESULTS_DIR,
                # storage_path=RESULTS_DIR,
                name=NAME,
                checkpoint_config=ray.air.CheckpointConfig(
                    checkpoint_frequency=5,
                    num_to_keep=5,
                    checkpoint_score_attribute="episode_reward_mean",
                    checkpoint_score_order="max"
                ),
            )
        ).fit()
    ```



4. **Compare results**

    After running the benchmarks, you can compare the results by examining the output files in the `results/` directory. These files include detailed metrics on energy consumption, carbon footprint, and workload distribution across data centers. Use these metrics to assess the relative performance of different algorithms and configurations.

### Evaluation Metrics

Green-DCC provides a range of evaluation metrics to assess the performance of the benchmarked algorithms:

- **Energy Consumption**: Total energy consumed by the data centers during the simulation.
- **Carbon Footprint**: Total carbon emissions generated by the data centers, calculated based on the energy mix and carbon intensity of the power grid.
- **Workload Distribution**: Efficiency of workload distribution across data centers, considering factors like latency, bandwidth cost, and data center utilization.

These metrics provide a comprehensive view of the performance of different algorithms and configurations, enabling you to identify the most effective strategies for sustainable data center management.

### Customizing Benchmarks

Green-DCC is designed to be highly customizable, allowing you to tailor the benchmark environment to your specific needs. You can modify the configuration files to:

- Add or remove data center locations.
- Adjust the workload characteristics, such as the proportion of shiftable tasks.
- Change the parameters of the RL algorithms, such as learning rates and discount factors.
- Include additional control strategies, such as energy storage or renewable energy integration.

Refer to the detailed documentation for more information on customizing the Green-DCC environment and running advanced benchmarks.

## Experimental Details

For all experiments, we considered three different locations: New York (NY), Atlanta (GA), and San Jose (CA). These locations were chosen to present a variety of weather conditions and carbon intensity profiles, creating a comprehensive and challenging evaluation environment. The goal was to develop a policy capable of addressing the unique challenges specific to each location. We utilized weather and carbon intensity data from the month of July. Weather data was sourced from [EnergyPlus](https://energyplus.net/weather), and carbon intensity data was retrieved from the [EIA API](https://api.eia.gov/bulk/EBA.zip). The base workload for our experiments was derived from open-source workload traces provided by Alibaba ([GitHub repository](https://github.com/alibaba/clusterdata)). Users can use their own data for weather, carbon intensity, and workload.

Each data center (DC) had a capacity of 1 Mega-Watt.

Green-DCC offers support for more locations beyond the three selected for these experiments. Detailed information about these additional locations can be found in the [Selected Locations](#selected-locations) section. The diverse climate and carbon intensity characteristics of these locations allow for extensive benchmarking and evaluation of RL controllers.

 **Weather and Carbon Intensity Data**

![Weather Data](Figures/GreenDCC_temperature.png)

*Figure Weather conditions (temperature) for New York, Atlanta, and San Jose over the month of July.*

![Carbon Intensity Data](Figures/GreenDCC_carbon_intensity.png)

*Figure Carbon intensity profiles for New York, Atlanta, and San Jose over the month of July.*

**Workload Distribution Comparison**

![Workload Distribution](Figures/GreenDCC_workload1.png)

*Figure Comparison of workload distribution across the three data centers under the Do Nothing controller.*

![Workload Distribution](Figures/GreenDCC_workload2.png)

*Figure Comparison of workload distribution across the three data centers under the HLO RL Controller.*


## Selected Locations

Green-DCC supports a wide range of locations, each with distinct weather patterns and carbon intensity profiles. This diversity allows for extensive benchmarking and evaluation of RL controllers under various environmental conditions. The table below provides a summary of the selected locations, including typical weather conditions and carbon emissions characteristics.

### Location Summaries

| Location   | Typical Weather                         | Carbon Intensity (CI)   |
|------------|-----------------------------------------|-------------------------|
| Arizona    | Hot, dry summers; mild winters          | High avg CI             |
| California | Mild, Mediterranean climate             | Medium avg CI           |
| Georgia    | Hot, humid summers; mild winters        | High avg CI             |
| Illinois   | Cold winters; hot, humid summers        | High avg CI             |
| New York   | Cold winters; hot, humid summers        | Medium avg CI           |
| Texas      | Hot summers; mild winters               | Medium avg CI           |
| Virginia   | Mild climate, seasonal variations       | Medium avg CI           |
| Washington | Mild, temperate climate; wet winters    | Low avg CI              |

*Table: Summary of Selected Locations with Typical Weather and Carbon Intensity Characteristics*

These locations were chosen because they are typical data center locations within the United States, offering a variety of environmental conditions that reflect real-world challenges faced by data centers.




## Contributing
Contributions are welcome. For major changes, please open an issue first to discuss what you would like to change. Please ensure to update tests as appropriate.

## Contact
For any questions or concerns, please open an issue in this repository.
