import os
import sys
from datetime import datetime
import socket
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp

file_dir = os.path.dirname(__file__)
# append one level upper directory to python path
sys.path.append(file_dir + '/..')
from HRL_withcooling.hierarchical_ppo import HierarchicalPPO as HRLPPO  # pylint: disable=C0413,E0611,E0001
from HRL_withcooling.greendcc_env import GreenDCC_Env  # pylint: disable=C0413

# pylint: disable=C0301,C0303,C0103,C0209
def main():

    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    env_name = "GreenDCC_Env"  # environment name

    hl_has_continuous_action_space = True  # continuous action space
    ll_has_continuous_action_space = True  # continuous action space
    dc_has_continuous_action_space = False  # discrete action space

    max_ep_len = 96*30                   # max timesteps in one episode
    max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.001                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ################ Hierarchical PPO hyperparameters ################
    update_timestep = max_ep_len      # update policy every n timesteps
    hl_K_epochs = 5               # update policy for K epochs in one PPO update for high level network
    ll_K_epochs = 5               # update policy for K epochs in one PPO update for low level network
    dc_K_epochs = 5               # update policy for K epochs in one PPO update for DC network

    eps_clip = 0.25             # clip parameter for PPO
    hl_gamma = 0.90            # discount factor for high level network
    ll_gamma = 0.99            # discount factor for low level network
    dc_gamma = 0.99            # discount factor for DC network

    hl_lr_actor = 0.00003       # learning rate for high level actor network
    hl_lr_critic = 0.0001       # learning rate for high level critic network
    ll_lr_actor = 0.00003       # learning rate for low level actor network(s)
    ll_lr_critic = 0.0001       # learning rate for low level critic network(s)
    dc_lr_actor = 0.00003       # learning rate for DC actor network(s)
    dc_lr_critic = 0.0001       # learning rate for DC critic network(s)

    random_seed = 43         # set random seed if required (0 = no random seed)
    #####################################################

    print("training environment name : " + env_name)
    env = GreenDCC_Env()

    # Include the HL agent 
    obs_space_hl = env.observation_space_hl
    action_space_hl = env.action_space_hl

    # Include the LL agent
    obs_space_ll = env.observation_space_ll
    action_space_ll = env.action_space_ll
    goal_dim_ll = env.goal_dimension_ll
    num_ll_policies = len(action_space_ll)

    # Include the DC agent
    obs_space_dc = env.observation_space_dc
    action_space_dc = env.action_space_dc
    goal_dim_dc = env.goal_dimension_dc
    num_dc_policies = len(action_space_dc)
    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "HRL_PPO_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # check action randomness
    #### get number of log files in log directory
    run_num = 10
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

    # Update the checkpoint paths to include the timestamp
    hl_checkpoint_path = directory + f"HL_PPO_{env_name}_{random_seed}_{run_num_pretrained}_{current_time_str}.pth"
    ll_checkpoint_path = directory + f"LL_PPO_{env_name}_{random_seed}_{run_num_pretrained}_{current_time_str}.pth"
    dc_checkpoint_path = directory + f"DC_PPO_{env_name}_{random_seed}_{run_num_pretrained}_{current_time_str}.pth"

    print(f"top level policy save checkpoint path : {hl_checkpoint_path}")
    print(f"low level policy save checkpoint path (partial) : {ll_checkpoint_path}")
    print(f"dc policy save checkpoint path (partial) : {dc_checkpoint_path}")
    #####################################################

    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension hl policy: ", obs_space_hl)
    print("state space dimension ll policy: ", obs_space_ll)
    print("state space dimension dc policy: ", obs_space_dc)
    print("goal dimension ll: ", goal_dim_ll)
    print("goal dimension dc: ", goal_dim_dc)
    print("action space dimension hl policy: ", action_space_hl)
    print("action space dimension ll policy: ", action_space_ll)
    print("action space dimension dc policy: ", action_space_dc)
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
    if dc_has_continuous_action_space:
        print("Initializing a continuous action space policy for DC agent(s)")
    else:
        print("Initializing a discrete action space DC agent(s)")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs top level : ", hl_K_epochs)
    print("PPO K epochs low level : ", ll_K_epochs)
    print("PPO K epochs dc : ", dc_K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) top level: ", hl_gamma)
    print("discount factor (gamma) low level: ", ll_gamma)
    print("discount factor (gamma) dc: ", dc_gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", hl_lr_actor, ll_lr_actor, dc_lr_actor)
    print("optimizer learning rate critic : ",hl_lr_critic, ll_lr_critic, dc_lr_critic)

    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)  # pylint: disable=no-member
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agents
    ppo_agent = HRLPPO(
        num_ll_policies=num_ll_policies,
        num_dc_policies=num_dc_policies,
        obs_dim_hl=obs_space_hl.shape[0],
        obs_dim_ll=[i.shape[0] for i in obs_space_ll],
        obs_dim_dc=[i.shape[0] for i in obs_space_dc],
        action_dim_hl=action_space_hl.shape[0],
        action_dim_ll=[i.shape[0] for i in action_space_ll],
        action_dim_dc=[i.n for i in action_space_dc],  # Use .n for discrete action spaces
        goal_dim_ll=[i for i in goal_dim_ll],
        goal_dim_dc=[i for i in goal_dim_dc],
        hl_lr_actor=hl_lr_actor,
        hl_lr_critic=hl_lr_critic,
        ll_lr_actor=ll_lr_actor,
        ll_lr_critic=ll_lr_critic,
        dc_lr_actor=dc_lr_actor,
        dc_lr_critic=dc_lr_critic,
        hl_gamma=hl_gamma,
        ll_gamma=ll_gamma,
        dc_gamma=dc_gamma,
        hl_K_epochs=hl_K_epochs,
        ll_K_epochs=ll_K_epochs,
        dc_K_epochs=dc_K_epochs,
        eps_clip=eps_clip,
        hl_has_continuous_action_space=hl_has_continuous_action_space,
        ll_has_continuous_action_space=ll_has_continuous_action_space,
        dc_has_continuous_action_space=dc_has_continuous_action_space,
        action_std_init=action_std,
        high_policy_action_freq=1,
        ll_policy_ids=env.datacenter_ids
    )

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    ################# tensorboard logging ################
    # current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    hostname = socket.gethostname()
    log_dir = f'HRL_runs/{current_time_str}_{hostname}'
    writer = SummaryWriter(log_dir)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,total_reward\n')

    # Initialize variables for tracking best reward
    best_total_reward = float('-inf')
    total_rewards = []
    num_episodes_for_checkpoint = 10  # Number of episodes to consider for checkpointing

    time_step = 0
    i_episode = 0

    # hierarchical training loop
    while time_step <= max_training_timesteps:
        state = env.reset()
        hl_current_ep_reward = 0
        ll_current_ep_reward = [0 for _ in range(num_ll_policies)]
        dc_current_ep_reward = [0 for _ in range(num_dc_policies)]

        # env specific traces for each episode
        CO2_footprint_per_step = []
        bat_total_energy_with_battery_KWh_per_step = []
        ls_tasks_dropped_per_step = []
        start_time = datetime.now().replace(microsecond=0)

        for _ in range(1, max_ep_len + 1):

            # select hierarchical actions
            action = ppo_agent.select_action(state)
            state, reward, done, info = env.step(action)

            # High-level policy
            if (ppo_agent.action_counter - 1) % ppo_agent.high_policy_action_freq == 0:
                ppo_agent.high_policy.buffer.rewards.append(reward['high_level_rewards'])
                ppo_agent.high_policy.buffer.is_terminals.append(done)

            # Low-level LS policies
            for i in range(num_ll_policies):
                ppo_agent.low_policies[i].buffer.rewards.append(reward[f'low_level_rewards_DC{i+1}'])
                ppo_agent.low_policies[i].buffer.is_terminals.append(done)

            # DC agent policies
            for i in range(num_dc_policies):
                ppo_agent.dc_policies[i].buffer.rewards.append(reward[f'dc_rewards_DC{i+1}'])
                ppo_agent.dc_policies[i].buffer.is_terminals.append(done)

            # Update rewards
            hl_current_ep_reward += reward['high_level_rewards']
            for i in range(num_ll_policies):
                ll_current_ep_reward[i] += reward[f'low_level_rewards_DC{i+1}']
            for i in range(num_dc_policies):
                dc_current_ep_reward[i] += reward[f'dc_rewards_DC{i+1}']

            time_step +=1

            # update PPO agent(s)
            if time_step % update_timestep == 0:
                ppo_loss = ppo_agent.update()

                # Group the high level policy loss under 'Loss/HighLevelPolicy'
                writer.add_scalar('Loss/HighLevelPolicy', ppo_loss[0], time_step)

                # Group the low level policy losses under 'Loss/LowLevelPolicy'
                for i in range(num_ll_policies):
                    writer.add_scalar(f'Loss/LowLevelPolicy/DC_{i+1}', ppo_loss[i+1], time_step)

                # Group the data center policy losses under 'Loss/DcPolicy'
                for i in range(num_dc_policies):
                    writer.add_scalar(f'Loss/DcPolicy/DC_{i+1}', ppo_loss[num_ll_policies+i+1], time_step)


            # if continuous action space; then decay action std of output action distribution
            if hl_has_continuous_action_space and ll_has_continuous_action_space and (time_step % action_std_decay_freq) == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # collect energy and temperature data from info dictionary
            CO2_footprint_per_step.append([info[f'low_level_info_{i}']['CO2_footprint_per_step'] for i in env.datacenter_ids])
            bat_total_energy_with_battery_KWh_per_step.append([info[f'low_level_info_{i}']['bat_total_energy_with_battery_KWh'] for i in env.datacenter_ids])
            ls_tasks_dropped_per_step.append([info[f'low_level_info_{i}']['ls_overdue_penalty'] for i in env.datacenter_ids])

            # break; if the episode is over
            done = done['high_level_done']
            if done:
                # Calculate the time needed to complete this episode
                end_time = datetime.now().replace(microsecond=0)
                
                print(f'Episode {i_episode} done at timestep {time_step} after {end_time - start_time}')
                start_time = end_time
                break

        # Update episode count
        i_episode += 1

        # Compute total rewards
        total_episode_reward = hl_current_ep_reward + sum(ll_current_ep_reward) + sum(dc_current_ep_reward)
        total_rewards.append(total_episode_reward)

        # Log to file
        log_f.write('{},{},{}\n'.format(i_episode, time_step, total_episode_reward))
        log_f.flush()

        # Log to TensorBoard
        writer.add_scalar('Rewards/HighLevelPolicy', hl_current_ep_reward, i_episode)
        for i in range(num_ll_policies):
            writer.add_scalar(f'Rewards/LowLevelPolicy/DC_{i+1}', ll_current_ep_reward[i], i_episode)

        for i in range(num_dc_policies):
            writer.add_scalar(f'Rewards/DcPolicy/DC_{i+1}', dc_current_ep_reward[i], i_episode)

        writer.add_scalar('Rewards/TotalEpisodeReward', total_episode_reward, i_episode)

        CO2_footprint_per_DC = np.sum(np.array(CO2_footprint_per_step), axis=0)
        bat_total_energy_with_battery_KWh_per_DC = np.sum(np.array(bat_total_energy_with_battery_KWh_per_step), axis=0)
        ls_tasks_dropped_per_DC = np.sum(np.array(ls_tasks_dropped_per_step), axis=0)

        for idx, i in enumerate(env.datacenter_ids):
            writer.add_scalar(f'Environment/CO2_Footprint/DC_{idx+1}', CO2_footprint_per_DC[idx], i_episode)
            writer.add_scalar(f'Environment/TotalEnergy/DC_{idx+1}', bat_total_energy_with_battery_KWh_per_DC[idx], i_episode)
            writer.add_scalar(f'Environment/TasksOverdue/DC_{idx+1}', ls_tasks_dropped_per_DC[idx], i_episode)

        # Print average total reward over last N episodes
        if len(total_rewards) >= num_episodes_for_checkpoint:
            average_total_reward = np.mean(total_rewards[-num_episodes_for_checkpoint:])
            print("Episode : {} \t Timestep : {} \t Average Total Reward over last {} episodes: {:.2f}".format(
                i_episode, time_step, num_episodes_for_checkpoint, average_total_reward))

            # Checkpoint saving logic
            accumulated_reward = sum(total_rewards[-num_episodes_for_checkpoint:])
            if accumulated_reward > best_total_reward:
                print("--------------------------------------------------------------------------------------------")
                print("New best accumulated reward over last {} episodes: {:.2f}".format(num_episodes_for_checkpoint, accumulated_reward))
                print(f"Saving models to checkpoints: {hl_checkpoint_path}, {ll_checkpoint_path}, {dc_checkpoint_path}")
                ppo_agent.save(hl_checkpoint_path, ll_checkpoint_path, dc_checkpoint_path)
                print("Models saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")
                best_total_reward = accumulated_reward

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at: ", start_time)
    print("Finished training at: ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    # Set the start method to 'spawn' to prevent CUDA re-initialization issues
    # mp.set_start_method('spawn', force=True)
    
    main()
