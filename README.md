# Green-DCC

## Dashboard

To watch the video, click on the screenshot below (right-click and select "Open link in new tab" to view in a new tab):

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
Green-DCC is a framework for testing multi-agent Reinforcement Learning (MARL) algorithms that optimizes data center clusters in a hierarchical manner for multiple objectives of carbon footprint reduction, energy consumption, and energy cost. It uses OpenAI Gym standard and supports modeling and control of three different types of problems: Carbon aware flexible load shifting, Data center HVAC cooling energy optimization and carbon aware battery auxiliary supply.

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
Refer to the [docs](https://hewlettpackard.github.io/green-dcc/) for documentation of Green-DCC.

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
