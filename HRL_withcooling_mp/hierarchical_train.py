import os
import sys
from datetime import datetime
import socket
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

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

    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 5           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e3)          # save model frequency (in num timesteps)
    # best_reward = float('-inf')       # initialize best reward as negative infinity
    print_avg_reward = 0                # initialize average reward

    action_std = 0.2                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.01                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ Hierarchical PPO hyperparameters ################
    update_timestep = max_ep_len // 4      # update policy every n timesteps
    # pylint : disable=C0103
    hl_K_epochs = 5               # update policy for K epochs in one PPO update for high level network
    ll_K_epochs = 5               # update policy for K epochs in one PPO update for low level network
    dc_K_epochs = 5               # update policy for K epochs in one PPO update for DC network

    eps_clip = 0.2             # clip parameter for PPO
    hl_gamma = 0.90            # discount factor for high level network
    ll_gamma = 0.99            # discount factor for low level network
    dc_gamma = 0.99            # discount factor for DC network

    hl_lr_actor = 0.0003       # learning rate for high level actor network
    hl_lr_critic = 0.001       # learning rate for high level critic network
    ll_lr_actor = 0.0003       # learning rate for low level actor network(s)
    ll_lr_critic = 0.001       # learning rate for low level critic network(s)
    dc_lr_actor = 0.0003       # learning rate for DC actor network(s)
    dc_lr_critic = 0.001       # learning rate for DC critic network(s)

    random_seed = 42         # set random seed if required (0 = no random seed)
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


    hl_checkpoint_path = directory + "HL_PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    ll_checkpoint_path = directory + "LL_PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    dc_checkpoint_path = directory + "DC_PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("top level policy save checkpoint path : " + hl_checkpoint_path)
    print("low level policy save checkpoint path (partial) : " + ll_checkpoint_path)
    print("dc policy save checkpoint path (partial) : " + dc_checkpoint_path)
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

    # initialize a PPO agent
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
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    hostname = socket.gethostname()
    log_dir = f'HRL_runs/{current_datetime}_{hostname}'
    writer = SummaryWriter(log_dir)
    
    print("============================================================================================")
    
    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')
    
    # printing and logging variables
    print_running_reward = [0 for i in range(1 + num_ll_policies)]
    print_running_episodes = 0

    log_running_reward = [0 for i in range(1 + num_ll_policies)]
    log_running_episodes = 0

    time_step = 0
    i_episode = 0
    
    best_reward = float('-inf')
    
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
                writer.add_scalar('high_level_policy_loss', ppo_loss[0], time_step)
                for i in range(num_ll_policies):
                    writer.add_scalar(f'low_level_policy_{i+1}_loss', ppo_loss[i+1], time_step)
                for i in range(num_dc_policies):
                    writer.add_scalar(f'dc_policy_{i+1}_loss', ppo_loss[num_ll_policies+i+1], time_step)
                    
            # if continuous action space; then decay action std of ouput action distribution
            if hl_has_continuous_action_space and ll_has_continuous_action_space and (time_step % action_std_decay_freq) == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)
                
            # log in logging file
            if time_step % log_freq == 0:
                # log average reward till last episode
                log_avg_reward = [i/log_running_episodes for i in log_running_reward]
                log_avg_reward = [round(i, 2) for i in log_avg_reward]

                log_f.write('{},{},'.format(i_episode, time_step) + ','.join(map(str, log_avg_reward)) + '\n')
                log_f.flush()

                log_running_reward = [0 for i in range(1 + num_ll_policies)]
                log_running_episodes = 0
            
            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = [i/print_running_episodes for i in print_running_reward]
                print_avg_reward = [round(i, 2) for i in print_avg_reward]

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward (HL + LL(s)) : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = [0 for i in range(1 + num_ll_policies)]
                print_running_episodes = 0
                
            # save model weights TODO: save model with best reward in list format
            if (time_step % save_model_freq) == 0:
                if np.sum(print_avg_reward) > best_reward:
                    print("--------------------------------------------------------------------------------------------")
                    print("saving top level model at : " + hl_checkpoint_path)
                    print("saving low level model at : " + ll_checkpoint_path)
                    print("saving dc model at : " + dc_checkpoint_path)
                    ppo_agent.save(hl_checkpoint_path, ll_checkpoint_path, dc_checkpoint_path)
                    print("models saved")
                    print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                    print("--------------------------------------------------------------------------------------------")
                    best_reward = np.sum(print_avg_reward)
                    
            # collect energy and temperature data from info dictionary
            CO2_footprint_per_step.append([info[f'low_level_info_{i}']['CO2_footprint_per_step'] for i in env.datacenter_ids])
            bat_total_energy_with_battery_KWh_per_step.append([info[f'low_level_info_{i}']['bat_total_energy_with_battery_KWh'] for i in env.datacenter_ids])
            ls_tasks_dropped_per_step.append([info[f'low_level_info_{i}']['ls_tasks_dropped'] for i in env.datacenter_ids])
            
            # break; if the episode is over
            done = done['high_level_done']
            if done:
                break
            
        writer.add_scalar('HL Policy Episode Reward', hl_current_ep_reward, i_episode) 
        for i in range(num_ll_policies):
            writer.add_scalar(f'LL Policy {i+1} Episode Reward', ll_current_ep_reward[i], i_episode)
        for i in range(num_dc_policies):
            writer.add_scalar(f'DC Policy {i+1} Episode Reward', dc_current_ep_reward[i], i_episode)
            
        CO2_footprint_per_DC = np.sum(np.array(CO2_footprint_per_step), axis=0)
        bat_total_energy_with_battery_KWh_per_DC = np.sum(np.array(bat_total_energy_with_battery_KWh_per_step), axis=0)
        ls_tasks_dropped_per_DC = np.sum(np.array(ls_tasks_dropped_per_step), axis=0)
        
        for idx,i in enumerate(env.datacenter_ids):
            writer.add_scalar(f'CO2_footprint {i}/Episode', CO2_footprint_per_DC[idx], i_episode)
            writer.add_scalar(f'bat_total_energy in kwh {i}/Episode', bat_total_energy_with_battery_KWh_per_DC[idx], i_episode)
            writer.add_scalar(f'ls_tasks_dropped {i}/Episode', ls_tasks_dropped_per_DC[idx], i_episode)
            # writer.add_scalar(f'Carbon Intensity {i}', info[f'low_level_info_{i}']['Carbon Intensity'], i_episode)
            # Add also the average tasks in queue

        print_running_reward = [i+j for i,j in zip(print_running_reward, [hl_current_ep_reward] + ll_current_ep_reward + dc_current_ep_reward)]
        print_running_episodes += 1

        log_running_reward = [i+j for i,j in zip(print_running_reward, [hl_current_ep_reward] + ll_current_ep_reward + dc_current_ep_reward)]
        log_running_episodes += 1 
        
        i_episode += 1
    
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

    main()
