import os
import sys
from datetime import datetime
import socket
import asyncio
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

file_dir = os.path.dirname(__file__)
# append one level upper directory to python path
sys.path.append(file_dir + '/..')
from HRL_multiple_rollouts.hierarchical_ppo import HierarchicalPPO as HRLPPO  # pylint: disable=C0413,E0611,E0001
from HRL_multiple_rollouts.greendcc_env import GreenDCC_Env  # pylint: disable=C0413

import multiprocessing as mp
from utils.utils_cf import generate_node_connections

def worker_process(worker_id, child_conn, env_config, ppo_agent_params, num_cores_per_worker, experience_per_worker):
    import os
    import torch
    import numpy as np
    from HRL_multiple_rollouts.hierarchical_ppo import HierarchicalPPO as HRLPPO
    from HRL_multiple_rollouts.greendcc_env import GreenDCC_Env

    # Get the available CPU cores and assign only a specific number per worker
    available_cores = list(range(os.cpu_count()))
    assigned_cores = available_cores[worker_id * num_cores_per_worker: (worker_id + 1) * num_cores_per_worker]
    pid = os.getpid()
    os.sched_setaffinity(pid, assigned_cores)  # Set the core affinity

    # Limit the number of threads for PyTorch and NumPy based on assigned cores
    torch.set_num_threads(num_cores_per_worker)
    torch.set_num_interop_threads(num_cores_per_worker)
    os.environ['OMP_NUM_THREADS'] = str(num_cores_per_worker)
    os.environ['MKL_NUM_THREADS'] = str(num_cores_per_worker)
    os.environ['NUMEXPR_NUM_THREADS'] = str(num_cores_per_worker)

    # Set random seed for this worker
    seed = 50 + env_config['random_seed'] + worker_id
    torch.manual_seed(seed)
    np.random.seed(seed)

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
        'low_policies': [
            {
                'states': [],
                'actions': [],
                'logprobs': [],
                'rewards': [],
                'state_values': [],
                'is_terminals': []
            } for _ in range(env_config['num_ll_policies'])
        ]
    }

    # state = env.reset()
    dones = {'high_level_done': False,
            'low_level_done_DC1': False,
            'low_level_done_DC2': False,
            'low_level_done_DC3': False}

    hl_current_ep_reward = 0
    ll_current_ep_reward = [0 for _ in range(env_config['num_ll_policies'])]
    episode_length = 0
    experience_counter = 0

    # Initialize lists to collect metrics
    CO2_footprint_per_step = []
    bat_total_energy_with_battery_KWh_per_step = []
    ls_tasks_dropped_per_step = []
    
    while experience_counter < experience_per_worker:
        # High-level action selection
        if ppo_agent.action_counter % ppo_agent.high_policy_action_freq == 0:
            high_level_action = ppo_agent.high_policy.select_action(state['high_level_obs'])
            ppo_agent.goal = high_level_action
            # Store high-level experiences
            experiences['high_policy']['states'].append(state['high_level_obs'].tolist())
            experiences['high_policy']['actions'].append(ppo_agent.high_policy.buffer.actions[-1].tolist())
            experiences['high_policy']['logprobs'].append(ppo_agent.high_policy.buffer.logprobs[-1].item())
            experiences['high_policy']['state_values'].append(ppo_agent.high_policy.buffer.state_values[-1].item())
            experiences['high_policy']['is_terminals'].append(dones['high_level_done'])
        ppo_agent.action_counter += 1

        # Prepare low-level goals from high-level action
        goal_list = []
        for _, edges in generate_node_connections(N=[i for i in range(len(ppo_agent.low_policies))], E=ppo_agent.goal).items():
            goal_list.append([e[1] for e in edges])

        actions = {'high_level_action': np.clip(ppo_agent.goal, -1.0, 1.0)}

        # Low-level action selection and experience storage
        for idx, (i, j, policy) in enumerate(zip(ppo_agent.ll_policy_ids, goal_list, ppo_agent.low_policies)):
            # Concatenate low-level observation with high-level action
            state_ll = np.concatenate([state['low_level_obs_' + i], j])

            # Select action
            low_level_action = np.clip(policy.select_action(state_ll), -1.0, 1.0)
            actions['low_level_action_' + i] = low_level_action

            # Store experiences
            experiences['low_policies'][idx]['states'].append(state_ll.tolist())
            experiences['low_policies'][idx]['actions'].append(policy.buffer.actions[-1].tolist())
            experiences['low_policies'][idx]['logprobs'].append(policy.buffer.logprobs[-1].item())
            experiences['low_policies'][idx]['state_values'].append(policy.buffer.state_values[-1].item())
            experiences['low_policies'][idx]['is_terminals'].append(dones[f'low_level_done_DC{idx+1}'])

        # Step the environment
        next_state, reward, dones, info = env.step(actions)
        episode_length += 1

        # Update rewards in experiences
        # High-level policy
        if (ppo_agent.action_counter - 1) % ppo_agent.high_policy_action_freq == 0:
            ppo_agent.high_policy.buffer.rewards.append(reward['high_level_rewards'])
            hl_current_ep_reward += reward['high_level_rewards']
            experiences['high_policy']['rewards'].append(reward['high_level_rewards'])
            experiences['high_policy']['is_terminals'][-1] = dones['high_level_done']

        # Low-level policies
        for idx in range(env_config['num_ll_policies']):
            ll_reward = reward[f'low_level_rewards_DC{idx+1}']
            ppo_agent.low_policies[idx].buffer.rewards.append(ll_reward)
            ll_current_ep_reward[idx] += ll_reward
            experiences['low_policies'][idx]['rewards'].append(ll_reward)
            experiences['low_policies'][idx]['is_terminals'][-1] = dones[f'low_level_done_DC{idx+1}']

        # Collect environmental metrics from the 'info' dictionary
        CO2_footprint_per_step.append([info[f'low_level_info_{i}']['CO2_footprint_per_step'] for i in env.datacenter_ids])
        bat_total_energy_with_battery_KWh_per_step.append([info[f'low_level_info_{i}']['bat_total_energy_with_battery_KWh'] for i in env.datacenter_ids])
        ls_tasks_dropped_per_step.append([info[f'low_level_info_{i}']['ls_overdue_penalty'] for i in env.datacenter_ids])

        # Check if the episode ended. If so, reset the environment.
        if dones['high_level_done']:
            state = env.reset()
        else:
            state = next_state

        experience_counter += 1
        # Stop collecting experiences if the target is reached
        # if experience_counter >= experience_count:
    # print(f'Worker completed an episode after {episode_length} timesteps')
            # break
        
        # if dones['high_level_done']:
        #     break

    # Calculate discounted rewards for high-level policy
    discounted_hl_rewards = []
    hl_discounted_reward = 0
    for reward, is_terminal in zip(reversed(experiences['high_policy']['rewards']),
                                   reversed(experiences['high_policy']['is_terminals'])):
        # if is_terminal:
            # hl_discounted_reward = 0
        hl_discounted_reward = reward + (ppo_agent.high_policy.gamma * hl_discounted_reward)
        discounted_hl_rewards.insert(0, hl_discounted_reward)
    experiences['high_policy']['rewards'] = discounted_hl_rewards

    # Calculate discounted rewards for low-level policies
    for i in range(env_config['num_ll_policies']):
        discounted_ll_rewards = []
        ll_discounted_reward = 0
        for reward, is_terminal in zip(reversed(experiences['low_policies'][i]['rewards']),
                                       reversed(experiences['low_policies'][i]['is_terminals'])):
            # if is_terminal:
                # ll_discounted_reward = 0
            ll_discounted_reward = reward + (ppo_agent.low_policies[i].gamma * ll_discounted_reward)
            discounted_ll_rewards.insert(0, ll_discounted_reward)
        experiences['low_policies'][i]['rewards'] = discounted_ll_rewards

    # Calculate per-DC metrics
    CO2_footprint_per_DC = np.sum(np.array(CO2_footprint_per_step), axis=0)
    bat_total_energy_with_battery_KWh_per_DC = np.sum(np.array(bat_total_energy_with_battery_KWh_per_step), axis=0)
    ls_tasks_dropped_per_DC = np.sum(np.array(ls_tasks_dropped_per_step), axis=0)

    # Include metrics in stats
    stats = {
        'hl_current_ep_reward': hl_current_ep_reward,
        'll_current_ep_reward': ll_current_ep_reward,
        'total_episode_reward': hl_current_ep_reward + sum(ll_current_ep_reward),
        'episode_length': episode_length,
        'CO2_footprint_per_DC': CO2_footprint_per_DC,
        'bat_total_energy_with_battery_KWh_per_DC': bat_total_energy_with_battery_KWh_per_DC,
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
        'low_policies': [
            {
                'states': [],
                'actions': [],
                'logprobs': [],
                'rewards': [],
                'state_values': [],
                'is_terminals': []
            } for _ in all_experiences[0]['low_policies']
        ]
    }

    for experience in all_experiences:
        # Aggregate high-level experiences
        for key in aggregated['high_policy']:
            aggregated['high_policy'][key].extend(experience['high_policy'][key])

        # Aggregate low-level experiences
        for i in range(len(aggregated['low_policies'])):
            for key in aggregated['low_policies'][i]:
                aggregated['low_policies'][i][key].extend(experience['low_policies'][i][key])

    return aggregated

# pylint: disable=C0301,C0303,C0103,C0209
def main():
    
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    env_name = "GreenDCC_Env"  # environment name

    hl_has_continuous_action_space = True  # continuous action space
    ll_has_continuous_action_space = True  # continuous action space

    max_ep_len = 96*7                   # max timesteps in one episode
    max_training_timesteps = int(10e6)   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 5           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e4)          # save model frequency (in num timesteps)
    # best_reward = float('-inf')       # initialize best reward as negative infinity
    print_avg_reward = 0                # initialize average reward

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.01                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ Hierarchical PPO hyperparameters ################
    update_timestep = max_ep_len      # update policy every n timesteps
    # pylint : disable=C0103
    hl_K_epochs = 5               # update policy for K epochs in one PPO update for high level network
    ll_K_epochs = 5               # update policy for K epochs in one PPO update for low level network

    eps_clip = 0.2             # clip parameter for PPO
    hl_gamma = 0.50            # discount factor for high level network
    ll_gamma = 0.99            # discount factor for low level network

    hl_lr_actor = 0.00003       # learning rate for high level actor network
    hl_lr_critic = 0.0001       # learning rate for high level critic network
    ll_lr_actor = 0.00003       # learning rate for low level actor network(s)
    ll_lr_critic = 0.0001       # learning rate for low level critic network(s)

    random_seed = 45         # set random seed if required (0 = no random seed)
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
    else:
        print("Initializing a discrete action space High Level policy")
    if ll_has_continuous_action_space:
        print("Initializing a continuous action space policy for low level agent(s)")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space Low Level policy/policies")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs top level : ", hl_K_epochs)
    print("PPO K epochs low level : ", ll_K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) top level: ", hl_gamma)
    print("discount factor (gamma) low level: ", ll_gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", hl_lr_actor, ll_lr_actor)
    print("optimizer learning rate critic : ",hl_lr_critic, ll_lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)  # pylint: disable=no-member
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")
    
    ################# training procedure ################

    # Define environment and agent configurations
    env_config = {
        'random_seed': random_seed,
        'num_ll_policies': num_ll_policies,
        'datacenter_ids': env.datacenter_ids
        # Add any other necessary configuration parameters
    }

    ppo_agent_params = {
        'num_ll_policies': num_ll_policies,
        'obs_dim_hl': obs_space_hl.shape[0],
        'obs_dim_ll': [i.shape[0] for i in obs_space_ll],
        'action_dim_hl': action_space_hl.shape[0],
        'action_dim_ll': [i.shape[0] for i in action_space_ll],
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
        'll_policy_ids': env.datacenter_ids
    }

    total_cores = os.cpu_count()
    num_workers = 4  # Example: you can adjust this dynamically
    num_cores_per_worker = 4  # Set how many cores each worker should use

    experience_per_worker = 128  # Experiences per worker
    experience_per_worker = max(96, experience_per_worker) # The experices per worker should be at least of 96 time steps (1 day)
    
    total_experiences = experience_per_worker * num_workers # The total number of experiences to collect
    
    print(f'Num workers: {num_workers}, Num cores per worker: {num_cores_per_worker}, Total cores: {total_cores}')
    print(f'Experiences per worker: {experience_per_worker}, Total experiences: {total_experiences}')


    workers = []
    parent_conns = []

    # Create worker processes
    for worker_id in range(num_workers):
        parent_conn, child_conn = mp.Pipe()
        worker = mp.Process(target=worker_process, args=(worker_id, child_conn, env_config, ppo_agent_params, num_cores_per_worker, experience_per_worker))
        worker.start()
        workers.append(worker)
        parent_conns.append(parent_conn)

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
    total_timesteps = 0
    i_episode = 0
    best_total_reward = float('-inf')
    total_rewards = []
    num_episodes_for_checkpoint = 10  # Number of episodes to consider for checkpointing
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    max_training_timesteps = int(3e6)  # Adjust as needed
    update_timestep = max_ep_len      # Update policy every n timesteps

    while total_timesteps <= max_training_timesteps:
        # Collect experiences from all workers
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
        ls_tasks_dropped_per_DC_all_workers = []
        
        for stats in stats_list:
            CO2_footprint_per_DC_all_workers.append(stats['CO2_footprint_per_DC'])
            bat_total_energy_with_battery_KWh_per_DC_all_workers.append(stats['bat_total_energy_with_battery_KWh_per_DC'])
            ls_tasks_dropped_per_DC_all_workers.append(stats['ls_tasks_dropped_per_DC'])

        # Convert lists to numpy arrays for aggregation
        CO2_footprint_per_DC_all_workers = np.array(CO2_footprint_per_DC_all_workers)
        bat_total_energy_with_battery_KWh_per_DC_all_workers = np.array(bat_total_energy_with_battery_KWh_per_DC_all_workers)
        ls_tasks_dropped_per_DC_all_workers = np.array(ls_tasks_dropped_per_DC_all_workers)

        # Aggregate metrics across all workers
        total_CO2_footprint_per_DC = np.sum(CO2_footprint_per_DC_all_workers, axis=0)
        total_bat_total_energy_with_battery_KWh_per_DC = np.sum(bat_total_energy_with_battery_KWh_per_DC_all_workers, axis=0)
        total_ls_tasks_dropped_per_DC = np.sum(ls_tasks_dropped_per_DC_all_workers, axis=0)

        # Log metrics to TensorBoard
        for idx, dc_id in enumerate(env.datacenter_ids):
            writer.add_scalar(f'Environment/CO2_Footprint/DC_{idx+1}', total_CO2_footprint_per_DC[idx], i_episode)
            writer.add_scalar(f'Environment/TotalEnergy/DC_{idx+1}', total_bat_total_energy_with_battery_KWh_per_DC[idx], i_episode)
            writer.add_scalar(f'Environment/TasksOverdue/DC_{idx+1}', total_ls_tasks_dropped_per_DC[idx], i_episode)

        # Aggregate experiences
        aggregated_experiences = aggregate_experiences(all_experiences)

        # Update global PPO agent with aggregated experiences
        # Calculate the time requiere to update the global PPO agent
        # curr_time = datetime.now().replace(microsecond=0)
        ppo_loss = global_ppo_agent.update_with_experiences(aggregated_experiences)
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
        # ppo_loss is a list: [high_policy_loss, low_policy_loss1, low_policy_loss2, ...]
        writer.add_scalar('Loss/HighLevelPolicy', ppo_loss[0], total_timesteps)
        for i in range(num_ll_policies):
            writer.add_scalar(f'Loss/LowLevelPolicy/DC_{i+1}', ppo_loss[i+1], total_timesteps)

        # Send updated policy parameters to workers
        policy_params = global_ppo_agent.get_policy_params()
        for conn in parent_conns:
            conn.send('update_policy')
            conn.send(policy_params)

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

    # Close worker processes
    for conn in parent_conns:
        conn.send('close')
    for worker in workers:
        worker.join()
        
if __name__ == '__main__':
    main()