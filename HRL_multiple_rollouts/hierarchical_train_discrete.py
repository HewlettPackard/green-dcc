import os
import re
import sys
from datetime import datetime
import time
import socket
import asyncio
import torch
import argparse
import subprocess

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

file_dir = os.path.dirname(__file__)
# append one level upper directory to python path
sys.path.append(file_dir + '/..')
from HRL_multiple_rollouts.hierarchical_ppo import HierarchicalPPO as HRLPPO  # pylint: disable=C0413,E0611,E0001
from HRL_multiple_rollouts.greendcc_env import GreenDCC_Env  # pylint: disable=C0413

import multiprocessing as mp
from utils.utils_cf import generate_node_connections
from tqdm import tqdm

def parse_lscpu_p():
    # Run the 'lscpu -p' command and capture its output
    try:
        lscpu_output = subprocess.check_output('lscpu -p', shell=True).decode('utf-8')
    except subprocess.CalledProcessError as e:
        print("Error executing 'lscpu -p' command.")
        raise e

    cpu_info = []
    for line in lscpu_output.strip().split('\n'):
        # Skip comments and empty lines
        if line.startswith('#') or not line.strip():
            continue
        # Remove any extra commas
        line = re.sub(r',+', ',', line.strip(','))
        parts = line.strip().split(',')
        if len(parts) < 4:
            continue  # Skip incomplete lines
        try:
            cpu_id = int(parts[0])
            core_id = int(parts[1])
            socket_id = int(parts[2])
        except ValueError:
            # Handle parsing errors
            continue
        cpu_info.append({
            'cpu_id': cpu_id,
            'core_id': core_id,
            'socket_id': socket_id
        })
    return cpu_info

def get_first_thread_per_core(cpu_info):
    core_to_cpus = defaultdict(list)
    for entry in cpu_info:
        core_key = (entry['socket_id'], entry['core_id'])
        cpu_id = entry['cpu_id']
        core_to_cpus[core_key].append(cpu_id)
    first_thread_cpus = []
    for core in sorted(core_to_cpus.keys()):
        logical_cpus = sorted(core_to_cpus[core])
        first_thread_cpu = logical_cpus[0]
        first_thread_cpus.append({
            'cpu_id': first_thread_cpu,
            'core_id': core[1],
            'socket_id': core[0]
        })
    return first_thread_cpus

def interleave_cores(first_thread_cpus):
    socket_0_cpus = [cpu for cpu in first_thread_cpus if cpu['socket_id'] == 0]
    socket_1_cpus = [cpu for cpu in first_thread_cpus if cpu['socket_id'] == 1]
    min_length = min(len(socket_0_cpus), len(socket_1_cpus))
    interleaved_cpus = []
    for i in range(min_length):
        interleaved_cpus.append(socket_0_cpus[i])
        interleaved_cpus.append(socket_1_cpus[i])
    if len(socket_0_cpus) > min_length:
        interleaved_cpus.extend(socket_0_cpus[min_length:])
    elif len(socket_1_cpus) > min_length:
        interleaved_cpus.extend(socket_1_cpus[min_length:])
    return interleaved_cpus

def assign_cores_to_workers(interleaved_cpus, num_workers, num_cores_per_worker, core_offset=0):
    assigned_cores = []
    total_physical_cores = len(interleaved_cpus)
    total_required_cores = num_workers * num_cores_per_worker

    if total_required_cores + core_offset > total_physical_cores:
        raise ValueError(
            f"Not enough physical cores ({total_physical_cores}) for {num_workers} workers with "
            f"{num_cores_per_worker} cores each and core_offset {core_offset}."
        )

    for worker_id in range(num_workers):
        start_idx = core_offset + worker_id * num_cores_per_worker
        end_idx = start_idx + num_cores_per_worker
        worker_cores = interleaved_cpus[start_idx:end_idx]
        assigned_cores.append([cpu['cpu_id'] for cpu in worker_cores])

    return assigned_cores

# def worker_process(worker_id, child_conn, env_config, ppo_agent_params, num_cores_per_worker, experience_per_worker, core_offset, num_workers):
def worker_process(worker_id, child_conn, env_config, ppo_agent_params, num_cores_per_worker, experience_per_worker, assigned_cores):

    import os
    import random
    import torch
    import numpy as np
    from HRL_multiple_rollouts.hierarchical_ppo import HierarchicalPPO as HRLPPO
    from HRL_multiple_rollouts.greendcc_env import GreenDCC_Env

    # # Get the available CPU cores and assign only a specific number per worker
    # total_cores = os.cpu_count()
    # available_cores = list(range(total_cores))

    # # Calculate start and end indices for core assignment
    # start_core = worker_id * num_cores_per_worker + core_offset
    # end_core = start_core + num_cores_per_worker

    # # Ensure we don't exceed the total number of cores
    # if end_core > total_cores:
    #     raise ValueError(f"Not enough CPU cores available for worker {worker_id} with core_offset {core_offset}")

    # assigned_cores = available_cores[start_core:end_core]
    # print(f'Worker {worker_id} with core_offset {core_offset} assigned cores: {assigned_cores}')

    # Get the available CPU cores and assign only a specific number per worker
    # available_cores = list(range(os.cpu_count()))
    # assigned_cores = available_cores[worker_id * num_cores_per_worker: (worker_id + 1) * num_cores_per_worker]
    pid = os.getpid()
    os.sched_setaffinity(pid, assigned_cores)  # Set the core affinity

    # Limit the number of threads for PyTorch and NumPy based on assigned cores
    # torch.set_num_threads(num_cores_per_worker)
    # torch.set_num_interop_threads(num_cores_per_worker)
    # os.environ['OMP_NUM_THREADS'] = str(num_cores_per_worker)
    # os.environ['MKL_NUM_THREADS'] = str(num_cores_per_worker)
    # os.environ['NUMEXPR_NUM_THREADS'] = str(num_cores_per_worker)

    # Limit the number of threads for PyTorch and NumPy
    num_threads = len(assigned_cores)
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)
    
    # Set random seed for this worker
    seed = env_config['random_seed'] + worker_id
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Initialize environment and PPO agent
    env = GreenDCC_Env()
    ppo_agent = HRLPPO(**ppo_agent_params)
    
    state = env.reset(seed=seed)

    while True:
        command = child_conn.recv()
        if command == 'collect_experience':
            experiences, stats, state = collect_experience(env, ppo_agent, env_config, state, experience_per_worker)
            child_conn.send((experiences, stats))
        elif command == 'update_policy':
            policy_params = child_conn.recv()
            ppo_agent.load_policy_params(policy_params)
        elif command == 'close':
            env.close()
            break


def collect_experience(env, ppo_agent, env_config, state, experience_per_worker):
    experiences = {
        'high_policy': {
            'states': [],
            'actions': [],
            'logprobs': [],
            'rewards': [],
            'state_values': [],
            'is_terminals': []
        },
        'low_policies': {
            dc_id: {
                'states': [],
                'actions': [],
                'logprobs': [],
                'rewards': [],
                'state_values': [],
                'is_terminals': []
            } for dc_id in ppo_agent.ll_policy_ids
        }
    }

    # Initialize dones for each datacenter
    dones = {'high_level_done': False}
    for dc_id in ppo_agent.ll_policy_ids:
        dones[f'low_level_done_{dc_id}'] = False

    hl_current_ep_reward = 0
    ll_current_ep_reward = {dc_id: 0 for dc_id in ppo_agent.ll_policy_ids}
    episode_length = 0
    experience_counter = 0

    # Initialize lists to collect metrics
    CO2_footprint_per_step = []
    bat_total_energy_with_battery_KWh_per_step = []
    ls_tasks_overdued_per_step = []
    ls_tasks_dropped_per_step = []
    
    while experience_counter < experience_per_worker:
        # High-level action selection
        if ppo_agent.action_counter % ppo_agent.high_policy_action_freq == 0:
            high_level_action, high_level_logprobs, high_level_state_values = ppo_agent.high_policy.select_action(state['high_level_obs'])
            ppo_agent.goal = high_level_action
            # Store high-level experiences
            experiences['high_policy']['states'].append(state['high_level_obs'].tolist())
            experiences['high_policy']['actions'].append([high_level_action.tolist()])
            experiences['high_policy']['logprobs'].append(high_level_logprobs)
            experiences['high_policy']['state_values'].append(high_level_state_values)
            experiences['high_policy']['is_terminals'].append(False)
        ppo_agent.action_counter += 1

        # Prepare low-level goals from high-level action
        node_connections = generate_node_connections(N=ppo_agent.ll_policy_ids, E=ppo_agent.goal)
        goal_mapping = {dc_id: [edge[1] for edge in edges] for dc_id, edges in node_connections.items()}

        actions = {'high_level_action': np.clip(ppo_agent.goal, -1.0, 1.0)}

        # Low-level action selection and experience storage
        for dc_id in ppo_agent.ll_policy_ids:
            policy = ppo_agent.low_policies[dc_id]
            goal = goal_mapping[dc_id]

            # Concatenate low-level observation with goal
            state_ll = np.concatenate([state['low_level_obs_' + dc_id], goal])

            # Select action
            low_level_action, low_level_logprobs, low_level_state_values = policy.select_action(state_ll)
            # low_level_action = np.clip(low_level_action, -1.0, 1.0)
            # Replace the low-level action with 1.0 to test the environment
            # low_level_action = np.array([1.0])
            actions['low_level_action_' + dc_id] = low_level_action

            # Store experiences
            experiences['low_policies'][dc_id]['states'].append(state_ll.tolist())
            experiences['low_policies'][dc_id]['actions'].append([low_level_action])
            experiences['low_policies'][dc_id]['logprobs'].append(low_level_logprobs)
            experiences['low_policies'][dc_id]['state_values'].append(low_level_state_values)
            experiences['low_policies'][dc_id]['is_terminals'].append(False)

        # Step the environment
        next_state, reward, dones, info = env.step(actions)
        episode_length += 1

        # Update rewards in experiences
        # High-level policy
        # if (ppo_agent.action_counter - 1) % ppo_agent.high_policy_action_freq == 0:
        # ppo_agent.high_policy.buffer.rewards.append(reward['high_level_rewards'])
        hl_current_ep_reward += reward['high_level_rewards']
        experiences['high_policy']['rewards'].append(reward['high_level_rewards'])
        # experiences['high_policy']['is_terminals'][-1] = False

        # Low-level policies
        for dc_id in ppo_agent.ll_policy_ids:
            policy = ppo_agent.low_policies[dc_id]
            ll_reward = reward[f'low_level_rewards_{dc_id}']
            # policy.buffer.rewards.append(ll_reward)
            ll_current_ep_reward[dc_id] += ll_reward
            experiences['low_policies'][dc_id]['rewards'].append(ll_reward)
            # experiences['low_policies'][dc_id]['is_terminals'][-1] = False

        # Collect environmental metrics from the 'info' dictionary
        CO2_footprint_per_step.append([info[f'low_level_info_{dc_id}']['CO2_footprint_per_step'] for dc_id in ppo_agent.ll_policy_ids])
        bat_total_energy_with_battery_KWh_per_step.append([info[f'low_level_info_{dc_id}']['bat_total_energy_with_battery_KWh'] for dc_id in ppo_agent.ll_policy_ids])
        ls_tasks_overdued_per_step.append([info[f'low_level_info_{dc_id}']['ls_overdue_penalty'] for dc_id in ppo_agent.ll_policy_ids])
        ls_tasks_dropped_per_step.append([info[f'low_level_info_{dc_id}']['ls_tasks_dropped'] for dc_id in ppo_agent.ll_policy_ids])

        # Check if the episode ended. If so, reset the environment.
        if dones['high_level_done']:
            state = env.reset()
        else:
            state = next_state

        experience_counter += 1

    # Calculate discounted rewards for high-level policy
    discounted_hl_rewards = []
    hl_discounted_reward = 0
    rewards = experiences['high_policy']['rewards'].copy()
    is_terminals = experiences['high_policy']['is_terminals']
    
    for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
        hl_discounted_reward = reward + (ppo_agent.high_policy.gamma * hl_discounted_reward)
        discounted_hl_rewards.insert(0, hl_discounted_reward)
    experiences['high_policy']['rewards'] = discounted_hl_rewards

    # Calculate discounted rewards for low-level policies
    for dc_id in ppo_agent.ll_policy_ids:
        policy = ppo_agent.low_policies[dc_id]
        rewards = experiences['low_policies'][dc_id]['rewards'].copy()
        is_terminals = experiences['low_policies'][dc_id]['is_terminals']

        discounted_ll_rewards = []
        ll_discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            ll_discounted_reward = reward + (policy.gamma * ll_discounted_reward)
            discounted_ll_rewards.insert(0, ll_discounted_reward)
        experiences['low_policies'][dc_id]['rewards'] = discounted_ll_rewards

    # Calculate per-DC metrics
    CO2_footprint_per_DC = np.sum(np.array(CO2_footprint_per_step), axis=0)
    bat_total_energy_with_battery_KWh_per_DC = np.sum(np.array(bat_total_energy_with_battery_KWh_per_step), axis=0)
    ls_tasks_overdued_per_DC = np.sum(np.array(ls_tasks_overdued_per_step), axis=0)
    ls_tasks_dropped_per_DC = np.sum(np.array(ls_tasks_dropped_per_step), axis=0)

    # Include metrics in stats
    stats = {
        'hl_current_ep_reward': hl_current_ep_reward,
        'll_current_ep_reward': list(ll_current_ep_reward.values()),
        'total_episode_reward': hl_current_ep_reward + sum(ll_current_ep_reward.values()),
        'episode_length': episode_length,
        'CO2_footprint_per_DC': CO2_footprint_per_DC,
        'bat_total_energy_with_battery_KWh_per_DC': bat_total_energy_with_battery_KWh_per_DC,
        'ls_tasks_overdued_per_DC': ls_tasks_overdued_per_DC,
        'ls_tasks_dropped_per_DC': ls_tasks_dropped_per_DC
    }

    return experiences, stats, state


def aggregate_experiences(all_experiences):
    aggregated = {
        'high_policy': {
            'states': [],
            'actions': [],
            'logprobs': [],
            'rewards': [],
            'state_values': [],
            'is_terminals': []
        },
        'low_policies': {
            dc_id: {
                'states': [],
                'actions': [],
                'logprobs': [],
                'rewards': [],
                'state_values': [],
                'is_terminals': []
            } for dc_id in all_experiences[0]['low_policies']
        }
    }

    for experience in all_experiences:
        # Aggregate high-level experiences
        for key in aggregated['high_policy']:
            aggregated['high_policy'][key].extend(experience['high_policy'][key])

        # Aggregate low-level experiences
        for dc_id in aggregated['low_policies']:
            for key in aggregated['low_policies'][dc_id]:
                aggregated['low_policies'][dc_id][key].extend(experience['low_policies'][dc_id][key])

    return aggregated


def evaluate_policy(env_config, ppo_agent, num_episodes=10, writer=None, total_timesteps=0):
    # Initialize the evaluation environment
    eval_env = GreenDCC_Env()
    state = eval_env.reset(seed=env_config['random_seed'])
    total_rewards = []
    # Initialize lists to collect metrics
    CO2_footprint_per_DC = []
    bat_total_energy_with_battery_KWh_per_DC = []
    ls_tasks_overdued_per_DC = []
    ls_tasks_dropped_per_DC = []

    for episode in range(num_episodes):
        done = False
        episode_reward = 0
        episode_CO2_footprint = []
        episode_bat_total_energy = []
        episode_tasks_overdued = []
        episode_tasks_dropped = []
        state = eval_env.reset()

        while not done:
            # High-level action selection (evaluation mode)
            high_level_action = ppo_agent.high_policy.select_action(state['high_level_obs'], evaluate=True)
            ppo_agent.goal = high_level_action

            # Prepare low-level goals from high-level action
            node_connections = generate_node_connections(N=ppo_agent.ll_policy_ids, E=ppo_agent.goal)
            goal_mapping = {dc_id: [edge[1] for edge in edges] for dc_id, edges in node_connections.items()}

            actions = {'high_level_action': np.clip(ppo_agent.goal, -1.0, 1.0)}

            # Low-level action selection (evaluation mode)
            for dc_id in ppo_agent.ll_policy_ids:
                policy = ppo_agent.low_policies[dc_id]
                goal = goal_mapping[dc_id]
                state_ll = np.concatenate([state['low_level_obs_' + dc_id], goal])
                low_level_action = policy.select_action(state_ll, evaluate=True)
                actions['low_level_action_' + dc_id] = low_level_action

            # Step the environment
            next_state, reward, dones, info = eval_env.step(actions)
            episode_reward += reward['high_level_rewards']
            episode_reward += sum(reward[f'low_level_rewards_{dc_id}'] for dc_id in ppo_agent.ll_policy_ids)

            # Collect metrics
            episode_CO2_footprint.append([info[f'low_level_info_{dc_id}']['CO2_footprint_per_step'] for dc_id in ppo_agent.ll_policy_ids])
            episode_bat_total_energy.append([info[f'low_level_info_{dc_id}']['bat_total_energy_with_battery_KWh'] for dc_id in ppo_agent.ll_policy_ids])
            episode_tasks_overdued.append([info[f'low_level_info_{dc_id}']['ls_overdue_penalty'] for dc_id in ppo_agent.ll_policy_ids])
            episode_tasks_dropped.append([info[f'low_level_info_{dc_id}']['ls_tasks_dropped'] for dc_id in ppo_agent.ll_policy_ids])

            if dones['high_level_done']:
                done = True
                # print(f'Evaluation episode {episode + 1} done. Total reward: {episode_reward}')
            else:
                state = next_state

        total_rewards.append(episode_reward)
        # Aggregate per-episode metrics
        CO2_footprint_per_DC.append(np.sum(np.array(episode_CO2_footprint), axis=0))
        bat_total_energy_with_battery_KWh_per_DC.append(np.sum(np.array(episode_bat_total_energy), axis=0))
        ls_tasks_overdued_per_DC.append(np.sum(np.array(episode_tasks_overdued), axis=0))
        ls_tasks_dropped_per_DC.append(np.sum(np.array(episode_tasks_dropped), axis=0))

    eval_env.close()
    # Compute average reward
    avg_reward = np.mean(total_rewards)

    # Log metrics if a writer is provided
    if writer is not None:
        writer.add_scalar('Evaluation/AverageReward', avg_reward, total_timesteps)
        for idx, dc_id in enumerate(ppo_agent.ll_policy_ids):
            avg_CO2 = np.mean([metrics[idx] for metrics in CO2_footprint_per_DC])
            avg_bat_energy = np.mean([metrics[idx] for metrics in bat_total_energy_with_battery_KWh_per_DC])
            avg_tasks_overdued = np.mean([metrics[idx] for metrics in ls_tasks_overdued_per_DC])
            avg_tasks_dropped = np.mean([metrics[idx] for metrics in ls_tasks_dropped_per_DC])

            writer.add_scalar(f'Evaluation/CO2_Footprint/DC_{idx+1}', avg_CO2, total_timesteps)
            writer.add_scalar(f'Evaluation/TotalEnergy/DC_{idx+1}', avg_bat_energy, total_timesteps)
            writer.add_scalar(f'Evaluation/TasksOverdue/DC_{idx+1}', avg_tasks_overdued, total_timesteps)
            writer.add_scalar(f'Evaluation/TasksDropped/DC_{idx+1}', avg_tasks_dropped, total_timesteps)

    return avg_reward


# pylint: disable=C0301,C0303,C0103,C0209
def main():

    parser = argparse.ArgumentParser(description='Hierarchical PPO Training Script')
    parser.add_argument('-c', '--core_offset', type=int, default=0, help='Core offset for CPU assignment')
    parser.add_argument('-n', '--num_workers', type=int, default=2, help='Number of worker processes')
    parser.add_argument('-e', '--experience_per_worker', type=int, default=96*3, help='Experiences per worker')
    parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed for environment')
    
    args = parser.parse_args()
    core_offset = args.core_offset
    num_workers = args.num_workers
    experience_per_worker = args.experience_per_worker
    random_seed = args.seed
    
    num_cores_per_worker = 1  # Set how many cores each worker should use
    
    # Parse lscpu -p output
    cpu_info = parse_lscpu_p()
    first_thread_cpus = get_first_thread_per_core(cpu_info)
    interleaved_cpus = interleave_cores(first_thread_cpus)
    assigned_cores_list = assign_cores_to_workers(interleaved_cpus, num_workers, num_cores_per_worker, core_offset=core_offset)

    for worker_id, cores in enumerate(assigned_cores_list):
        print(f'Run with core_offset {core_offset}: Worker {worker_id} assigned cores: {cores}')

    print("============================================================================================")

    total_cores = os.cpu_count()
    # num_workers = 8  # Example: you can adjust this dynamically

    # experience_per_worker = 96*3  # Experiences per worker
    # experience_per_worker = max(96, experience_per_worker) # The experices per worker should be at least of 96 time steps (1 day)
    
    total_experiences = experience_per_worker * num_workers # The total number of experiences to collect
    
    print(f'Num workers: {num_workers}, Num cores per worker: {num_cores_per_worker}, Total cores: {total_cores}')
    print(f'Experiences per worker: {experience_per_worker}, Total experiences: {total_experiences}')


    ####### initialize environment hyperparameters ######
    env_name = "GreenDCC_Env"  # environment name

    hl_has_continuous_action_space = True  # continuous action space
    ll_has_continuous_action_space = False  # continuous action space

    max_ep_len = 96*7                   # max timesteps in one episode
    max_training_timesteps = int(10e9)   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 5           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e4)          # save model frequency (in num timesteps)
    # best_reward = float('-inf')       # initialize best reward as negative infinity
    print_avg_reward = 0                # initialize average reward

    action_std = 0.5                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.02        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.01                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(100e3)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ Hierarchical PPO hyperparameters ################
    # update_timestep = max_ep_len // 4      # update policy every n timesteps
    # pylint : disable=C0103
    hl_K_epochs = 5               # update policy for K epochs in one PPO update for high level network
    ll_K_epochs = 5               # update policy for K epochs in one PPO update for low level network

    eps_clip = 0.3             # clip parameter for PPO
    hl_gamma = 0.50            # discount factor for high level network
    ll_gamma = 0.95            # discount factor for low level network

    hl_lr_actor = 0.0003       # learning rate for high level actor network
    hl_lr_critic = 0.001       # learning rate for high level critic network
    ll_lr_actor = 0.0003       # learning rate for low level actor network(s)
    ll_lr_critic = 0.001       # learning rate for low level critic network(s)

    # random_seed = 80         # set random seed if required (0 = no random seed)
    #####################################################

    print("training environment name : " + env_name)
    env = GreenDCC_Env()
    
    obs_space_hl = env.observation_space_hl
    action_space_hl = env.action_space_hl
    obs_space_ll = env.observation_space_ll
    action_space_ll = env.action_space_ll
    goal_dim_ll = env.goal_dimension_ll
    
    num_ll_policies = len(action_space_ll)

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "HRL_PPO_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 0     #### change this to prevent overwriting weights in same env_name folder

    directory = "HRL_PPO_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Add current datetime to checkpoint paths
    current_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    hl_checkpoint_path = directory + f"HL_PPO_{env_name}_{random_seed}_{run_num_pretrained}_{current_time_str}.pth"
    ll_checkpoint_path = directory + f"LL_PPO_{env_name}_{random_seed}_{run_num_pretrained}_{current_time_str}.pth"

    print("top level policy save checkpoint path : " + hl_checkpoint_path)
    print("low level policy save checkpoint path (partial) : " + ll_checkpoint_path)
    #####################################################
    
    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension hl policy: ", obs_space_hl)
    print("state space dimension ll policy: ", obs_space_ll)
    print("goal dimension : ", goal_dim_ll)
    print("action space dimension hl policy: ", action_space_hl)
    print("action space dimension ll policy: ", action_space_ll)
    print("--------------------------------------------------------------------------------------------")
    if hl_has_continuous_action_space:
        print("Initializing a continuous action space policy for top level agent")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
        print("--------------------------------------------------------------------------------------------")
    else:
        print("Initializing a discrete action space High Level policy")
        
    if ll_has_continuous_action_space:
        print("Initializing a continuous action space policy for low level agent(s)")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
        print("--------------------------------------------------------------------------------------------")
    else:
        print("Initializing a discrete action space Low Level policy/policies")
        
    # print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs top level : ", hl_K_epochs)
    print("PPO K epochs low level : ", ll_K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) top level: ", hl_gamma)
    print("discount factor (gamma) low level: ", ll_gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", hl_lr_actor, ll_lr_actor)
    print("optimizer learning rate critic : ",hl_lr_critic, ll_lr_critic)
    # if random_seed:
    #     print("--------------------------------------------------------------------------------------------")
    #     print("setting random seed to ", random_seed)
    #     torch.manual_seed(random_seed)
    #     env.seed(random_seed)  # pylint: disable=no-member
    #     np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")
    
    ################# training procedure ################

    datacenter_ids = sorted(env.datacenter_ids)
    # Define environment and agent configurations
    env_config = {
        'random_seed': random_seed,
        'num_ll_policies': num_ll_policies,
        'datacenter_ids': datacenter_ids
        # Add any other necessary configuration parameters
    }

    ppo_agent_params = {
        'num_ll_policies': num_ll_policies,
        'obs_dim_hl': obs_space_hl.shape[0],
        'obs_dim_ll': [i.shape[0] for i in obs_space_ll],
        'action_dim_hl': action_space_hl.shape[0],
        'action_dim_ll': [i.n for i in action_space_ll],  # Use .n for discrete action spaces
        'goal_dim_ll': [i for i in goal_dim_ll],
        'hl_lr_actor': hl_lr_actor,
        'hl_lr_critic': hl_lr_critic,
        'll_lr_actor': ll_lr_actor,
        'll_lr_critic': ll_lr_critic,
        'hl_gamma': hl_gamma,
        'll_gamma': ll_gamma,
        'hl_K_epochs': hl_K_epochs,
        'll_K_epochs': ll_K_epochs,
        'eps_clip': eps_clip,
        'hl_has_continuous_action_space': hl_has_continuous_action_space,
        'll_has_continuous_action_space': ll_has_continuous_action_space,
        'action_std_init': action_std,
        'high_policy_action_freq': 1,
        'll_policy_ids': datacenter_ids
    }


    workers = []
    parent_conns = []

    next_action_std_decay_step = action_std_decay_freq  # Initialize the next decay step
    
    for worker_id in range(num_workers):
        parent_conn, child_conn = mp.Pipe()
        worker_assigned_cores = assigned_cores_list[worker_id]
        worker = mp.Process(target=worker_process, args=(
            worker_id, child_conn, env_config, ppo_agent_params,
            num_cores_per_worker, experience_per_worker, worker_assigned_cores
        ))
        worker.start()
        workers.append(worker)
        parent_conns.append(parent_conn)
    # # Create worker processes
    # for worker_id in range(num_workers):
    #     parent_conn, child_conn = mp.Pipe()
    #     worker = mp.Process(target=worker_process, args=(worker_id, child_conn, env_config, ppo_agent_params, num_cores_per_worker, experience_per_worker, core_offset, num_workers))
    #     worker.start()
    #     workers.append(worker)
    #     parent_conns.append(parent_conn)

    # Initialize the global PPO agent
    global_ppo_agent = HRLPPO(**ppo_agent_params)


    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    ################# tensorboard logging ################
    # TensorBoard logging setup
    hostname = socket.gethostname()
    current_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'HRL_runs/{current_time_str}_{hostname}'
    writer = SummaryWriter(log_dir)

    # Logging variables
    # target_timesteps = 5000  # Example: track time for 100,000 timesteps

    total_timesteps = 0
    i_episode = 0
    best_total_reward = float('-inf')
    total_rewards = []
    num_episodes_for_checkpoint = 10  # Number of episodes to consider for checkpointing
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    max_training_timesteps = int(300e6)  # Adjust as needed
    # update_timestep = max_ep_len // 2     # Update policy every n timesteps
    
    evaluation_interval = 10000  # Evaluate every 100,000 timesteps
    next_evaluation_step = evaluation_interval

    # Record the start time
    benchmark_start_time = time.time()

    while total_timesteps <= max_training_timesteps:
        # Collect experiences from all workers
        try:
            for conn in parent_conns:
                conn.send('collect_experience')

            all_experiences = []
            stats_list = []
            for conn in parent_conns:
                experiences, stats = conn.recv()
                all_experiences.append(experiences)
                stats_list.append(stats)

            # print("All experiences:")
            # for exp in all_experiences:
            #     print(exp.keys())

            # print("Stats list:")
            # for stats in stats_list:
            #     print(stats.keys())

            # Aggregate metrics
            CO2_footprint_per_DC_all_workers = []
            bat_total_energy_with_battery_KWh_per_DC_all_workers = []
            ls_tasks_overdued_per_DC_all_workers = []
            ls_tasks_dropped_per_DC_all_workers = []
            
            for stats in stats_list:
                CO2_footprint_per_DC_all_workers.append(stats['CO2_footprint_per_DC'])
                bat_total_energy_with_battery_KWh_per_DC_all_workers.append(stats['bat_total_energy_with_battery_KWh_per_DC'])
                ls_tasks_overdued_per_DC_all_workers.append(stats['ls_tasks_overdued_per_DC'])
                ls_tasks_dropped_per_DC_all_workers.append(stats['ls_tasks_dropped_per_DC'])

            # Convert lists to numpy arrays for aggregation
            CO2_footprint_per_DC_all_workers = np.array(CO2_footprint_per_DC_all_workers)
            bat_total_energy_with_battery_KWh_per_DC_all_workers = np.array(bat_total_energy_with_battery_KWh_per_DC_all_workers)
            ls_tasks_overdued_per_DC_all_workers = np.array(ls_tasks_overdued_per_DC_all_workers)

            # Aggregate metrics across all workers
            total_CO2_footprint_per_DC = np.sum(CO2_footprint_per_DC_all_workers, axis=0)
            total_bat_total_energy_with_battery_KWh_per_DC = np.sum(bat_total_energy_with_battery_KWh_per_DC_all_workers, axis=0)
            total_ls_tasks_overdued_per_DC = np.sum(ls_tasks_overdued_per_DC_all_workers, axis=0)

            # Log metrics to TensorBoard
            for idx, dc_id in enumerate(datacenter_ids):
                writer.add_scalar(f'Environment/CO2_Footprint/DC_{idx+1}', total_CO2_footprint_per_DC[idx], i_episode)
                writer.add_scalar(f'Environment/TotalEnergy/DC_{idx+1}', total_bat_total_energy_with_battery_KWh_per_DC[idx], i_episode)
                writer.add_scalar(f'Environment/TasksOverdue/DC_{idx+1}', total_ls_tasks_overdued_per_DC[idx], i_episode)
                writer.add_scalar(f'Environment/TasksDropped/DC_{idx+1}', ls_tasks_dropped_per_DC_all_workers[0][idx], i_episode)

            # Log the current learning rates for actor and critic (both high-level and low-level policies)
            hl_actor_lr = global_ppo_agent.high_policy.scheduler_actor.get_last_lr()[0]
            hl_critic_lr = global_ppo_agent.high_policy.scheduler_critic.get_last_lr()[0]
            writer.add_scalar('LearningRate/HighLevelActor', hl_actor_lr, total_timesteps)
            writer.add_scalar('LearningRate/HighLevelCritic', hl_critic_lr, total_timesteps)

            # For low-level policies, assuming each has its own scheduler
            for i, dc_id in enumerate(global_ppo_agent.ll_policy_ids):
                ll_actor_lr = global_ppo_agent.low_policies[dc_id].scheduler_actor.get_last_lr()[0]
                ll_critic_lr = global_ppo_agent.low_policies[dc_id].scheduler_critic.get_last_lr()[0]
                writer.add_scalar(f'LearningRate/LowLevelActor/DC_{i+1}', ll_actor_lr, total_timesteps)
                writer.add_scalar(f'LearningRate/LowLevelCritic/DC_{i+1}', ll_critic_lr, total_timesteps)

            # Aggregate experiences
            aggregated_experiences = aggregate_experiences(all_experiences)

            # Update global PPO agent with aggregated experiences
            # Calculate the time requiere to update the global PPO agent
            # curr_time = datetime.now().replace(microsecond=0)
            high_policy_loss, low_policy_losses = global_ppo_agent.update_with_experiences(aggregated_experiences)
            # print("Time taken to update the global PPO agent: ", datetime.now().replace(microsecond=0) - curr_time)

            # Log losses and rewards
            # Compute total rewards from stats
            total_hl_rewards = sum([stats['hl_current_ep_reward'] for stats in stats_list])
            ll_total_rewards = [0 for _ in range(num_ll_policies)]
            for i in range(num_ll_policies):
                ll_total_rewards[i] = sum([stats['ll_current_ep_reward'][i] for stats in stats_list])

            total_episode_reward = sum([stats['total_episode_reward'] for stats in stats_list])
            total_rewards.extend([stats['total_episode_reward'] for stats in stats_list])
            total_episode_lengths = [stats['episode_length'] for stats in stats_list]

            # Log to TensorBoard
            avg_hl_reward = total_hl_rewards / num_workers
            writer.add_scalar('Rewards/HighLevelPolicy', avg_hl_reward, i_episode)
            for i in range(num_ll_policies):
                avg_ll_reward = ll_total_rewards[i] / num_workers
                writer.add_scalar(f'Rewards/LowLevelPolicy/DC_{i+1}', avg_ll_reward, i_episode)

            avg_total_reward = total_episode_reward / num_workers
            writer.add_scalar('Rewards/TotalEpisodeReward', avg_total_reward, i_episode)

            # Log losses
            # Log high-level policy losses (actor and critic separately)
            writer.add_scalar('Loss/HighLevelPolicy_Actor', high_policy_loss[0], total_timesteps) # Actor loss
            writer.add_scalar('Loss/HighLevelPolicy_Critic', high_policy_loss[1], total_timesteps) # Critic loss

            # Log low-level policy losses (actor and critic separately for each data center)
            for i, dc_id in enumerate(global_ppo_agent.ll_policy_ids):
                writer.add_scalar(f'Loss/LowLevelPolicy/DC_{i+1}_Actor', low_policy_losses[dc_id][0], total_timesteps)
                writer.add_scalar(f'Loss/LowLevelPolicy/DC_{i+1}_Critic', low_policy_losses[dc_id][1], total_timesteps)

            # Send updated policy parameters to workers
            policy_params = global_ppo_agent.get_policy_params()
            for conn in parent_conns:
                conn.send('update_policy')
                conn.send(policy_params)

            # Decay action_std if it's time
            if total_timesteps >= next_action_std_decay_step:
                global_ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)
                
                next_action_std_decay_step += action_std_decay_freq  # Schedule next decay step
                
                # Optionally log the new action_std for both high-level and low-level policies
                current_action_std = global_ppo_agent.high_policy.action_std
                print(f"Decayed action_std to {current_action_std}")
                writer.add_scalar('Hyperparameters/ActionStd_HighLevel', current_action_std, total_timesteps)
                # For low-level policies, you can log the action_std of one policy as an example
                ll_policy_id = list(global_ppo_agent.low_policies.keys())[0]
                # current_action_std_ll = global_ppo_agent.low_policies[ll_policy_id].action_std
                # writer.add_scalar(f'Hyperparameters/ActionStd_LowLevel_{ll_policy_id}', current_action_std_ll, total_timesteps)

            # if total_timesteps >= target_timesteps:
            #     end_time = time.time()  # Record the end time
            #     elapsed_time = end_time - benchmark_start_time  # Calculate the elapsed time
            #     print(f"Time required for {num_cores_per_worker} num workers per core is: {elapsed_time:.2f} seconds")
                # break

            total_timesteps += sum(total_episode_lengths)
            i_episode += num_workers

            # Save models periodically
            if len(total_rewards) >= num_episodes_for_checkpoint * num_workers:
                average_total_reward = np.mean(total_rewards[-num_episodes_for_checkpoint * num_workers:])
                
                print(f'Episode: {i_episode}, Timestep: {total_timesteps}, Average Total Reward over last {num_episodes_for_checkpoint * num_workers} episodes: {average_total_reward:.2f}, Best Total Reward: {best_total_reward:.2f}')

                # Checkpoint saving logic
                if average_total_reward > best_total_reward:
                    print("--------------------------------------------------------------------------------------------")
                    print("New best average reward over last {} episodes: {:.2f}".format(num_episodes_for_checkpoint * num_workers, average_total_reward))
                    print(f"Saving models to checkpoints: {hl_checkpoint_path}, {ll_checkpoint_path}")
                    global_ppo_agent.save(hl_checkpoint_path, ll_checkpoint_path)
                    print("Models saved")
                    elapsed_time = datetime.now().replace(microsecond=0) - start_time
                    print("Elapsed Time  : ", elapsed_time)
                    print("--------------------------------------------------------------------------------------------")
                    best_total_reward = average_total_reward
            
            if total_timesteps >= next_evaluation_step:
                avg_reward = evaluate_policy(env_config, global_ppo_agent, num_episodes=5, writer=writer, total_timesteps=total_timesteps)
                print(f'Evaluation at timestep {total_timesteps}: Average Reward: {avg_reward}')
                next_evaluation_step += evaluation_interval
        
        # Print the complete trace of the error
        except Exception as e:
            print("Error occurred: ", e)
            traceback.print_exc()
            continue
    # Close worker processes
    for conn in parent_conns:
        conn.send('close')
    for worker in workers:
        worker.join()
        
if __name__ == '__main__':
    main()