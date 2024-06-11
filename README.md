# Green-DCC: Benchmarking Dynamic Workload Distribution Techniques for Sustainable Data Center Cluster

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Benchmarking](#benchmarking)
5. [Experimental Details](#experimental-details)
6. [Additional Results](#additional-results)
7. [Selected Locations](#selected-locations)
8. [Conclusion](#conclusion)
9. [References](#references)

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

### Available Algorithms

Green-DCC supports benchmarking multiple heuristic and hierarchical reinforcement learning (RL) approaches. The following RL algorithms are included:

- **Advantage Actor-Critic (A2C)**
- **Asynchronous Advantage Actor-Critic (A3C)**
- **Adaptive Proximal Policy Optimization (APPO)**
- **Augmented Random Search (ARS)**
- **Deep Deterministic Policy Gradient (DDPG)**
- **Importance Weighted Actor-Learner Architecture (IMPALA)**
- **Proximal Policy Optimization (PPO)**
- **Soft Actor-Critic (SAC)**
- **Twin Delayed Deep Deterministic Policy Gradient (TD3)**
- **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)**

These algorithms can be trained and evaluated within the Green-DCC environment using Ray to compare their performance in terms of energy consumption, carbon footprint, and other relevant metrics. For more details on these algorithms, refer to the [Ray RLlib documentation](https://docs.ray.io/en/releases-2.4.0/rllib/rllib-algorithms.html).


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







# Green-DCC

## Dashboard

To watch the video, click on the screenshot below (right-click and select "Open link in new tab" to view in a new tab): #TODO go through/update links/content in links

[![Dashboard, click it to visualize it](media/DCRL_screenshot2.png)](https://www.dropbox.com/scl/fi/85gumlvjgbbk5kwjhee3i/Data-Center-Green-Dashboard-ver2.mp4?rlkey=w3mu21qqdk9asi826cjyyutzl&dl=0)

If you wish to download the video directly, [click here](https://www.dropbox.com/scl/fi/85gumlvjgbbk5kwjhee3i/Data-Center-Green-Dashboard-ver2.mp4?rlkey=w3mu21qqdk9asi826cjyyutzl&dl=1).


Demo of Green-DCC functionality
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XF92aR6nVYxENrviHeFyuRu0exKBb-nh?usp=sharing)
---

This repository contains the datasets and code for the paper "Green-DCC: Benchmarking Dynamic Workload Distribution Techniques for Sustainable Data Center Cluster".
---
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ldxlcG_prPw9U26alK9oRN2XvxrxNSDP?usp=sharing)


<p align="center">
  <img src="https://github.com/HewlettPackard/dc-rl/blob/main/sphinx/images/DCRL-sim1.png" align="centre" width="500" />
</p>

## Introduction
Green-DCC is a framework for testing multi-agent Reinforcement Learning (MARL) algorithms that optimizes data center clusters in a hierarchical manner for multiple objectives of carbon footprint reduction, energy consumption, and energy cost. It uses OpenAI Gym standard and supports modeling and control of three different types of problems: carbon aware flexible load shifting, data center HVAC cooling energy optimization and carbon aware battery auxiliary supply.

Main contributions of Green-DCC:

- the first OpenAI framework, to the best of our knowledge, focused on carbon footprint reduction for data center clusters
- support for hierarchical data center clusters modeling where users have flexibility in specifying the cluster architecture
- modular design meaning users can utilize pre-defined modules for load shifting, energy and battery or build their own 
- scalable architecture that allows multiple different types of modules and connections between them
- robust data center model that provides in-depth customization to fit users' needs 
- provides pre-defined reward functions as well as interface to create custom reward functions 
- built-in mechanisms for reward shaping focused on degree of cooperation between the agents and level of prioritization of carbon footprint reduction versus energy cost
- custom reward shaping through custom reward functions 
- build-in MARL algorithms, with ability to incorporate user-specified custom agents 


## Works Utilizing Green-DCC

#TODO

## Documentation and Installation
Refer to the [docs](https://hewlettpackard.github.io/green-dcc/) for documentation of Green-DCC. #TODO publish docs on GH Pages 

# Quick Start Guide

## Prerequisites
- Linux OS (Ubuntu 20.04)
- Conda


## Installation
First, download the repository. If using HTML, execute:
```bash
$ git clone https://github.com/HewlettPackard/green-dcc.git
```
If using SSH, execute:
```bash
$ git clone git@github.com:HewlettPackard/green-dcc.git
```
### Installing the Green-DCC environment 
Make sure you have conda installed. For more instructions on installing conda please check the [documentation](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html#install-linux-silent).

Change the current working directory to the green-dcc folder:

```bash
$ cd green-dcc
```

Create a conda environment and install dependencies:
```bash
$ conda create -n green-dcc python=3.10
$ conda activate green-dcc
$ pip install -r requirements.txt #TODO go through reqs 
```

## Usage
Before running the Green-DCC environment, make sure you are in the ```green-dcc``` folder. If you are in your home directory, run ```cd green-dcc``` or ```cd PATH_TO_PROJECT``` depending on where you downloaded the GitHub repository. 

### Running the Green-DCC environment with a random agent
To run an episode of the environment with a random agent execute:
```bash
$ python dcrl_env.py
```

### Training an RL agent on the Green-DCC environment
To start training, run the following command:

(Note: The `episode_reward_mean` will be `nan` for the first few iterations until 1 episode is completed)

For PPO:
```bash
$ python train_ppo.py
```

For MADDPG:
```bash
$ python train_maddpg.py
```

For A2C:
```bash
$ python train_a2c.py
```

### Running in Background Mode
If you want to run the Green-DCC framework in background mode use the following command:

```bash
$ nohup python PYTHON_SCRIPT > OUTPUT_FILE.txt  &
```
where ```PYTHON_SCRIPT``` is the script you want to run (e.g., ```train_ppo.py```) and ```OUTPUT_FILE``` is the name of the file that will contain the output (e.g. ```latest_experiment_output.txt```).

### Monitoring Training
Monitor the training using TensorBoard. By default, the location of the training data is at ```./results```. To visualize, run:

```bash
$ tensorboard --logdir ./results
```

## Contributing
Contributions are welcome. For major changes, please open an issue first to discuss what you would like to change. Please ensure to update tests as appropriate.

## Contact
For any questions or concerns, please open an issue in this repository.
