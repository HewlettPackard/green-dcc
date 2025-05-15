# train_hvac_agent.py
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import trange
import os
import yaml
import json
import datetime
import logging
import random
import pandas as pd
from gymnasium import spaces
from tqdm import trange, tqdm
from collections import deque

# --- Local Imports ---
# Adjust paths as necessary
from envs.sustaindc.dc_gym import dc_gymenv
# *** IMPORT THE ACTUAL DC_Config CLASS ***
from utils.dc_config_reader import DC_Config
# Agent networks and storage
from rl_components.agent_net import ActorNet, ValueNet
# Utility imports
from utils.managers import Time_Manager, Weather_Manager
from utils.config_logger import setup_logger
import envs.sustaindc.datacenter_model as DataCenter

# --- Rollout Storage (Adapted PPO version) ---
class PPORolloutStorage:
    # (Identical to the PPORolloutStorage class defined previously)
    def __init__(self, n_steps, obs_dim, device): self.n_steps = n_steps; self.obs_dim = obs_dim; self.device = device; self.reset()
    def reset(self): self.obs = np.zeros((self.n_steps, self.obs_dim), dtype=np.float32); self.actions = np.zeros(self.n_steps, dtype=np.int64); self.log_probs = np.zeros(self.n_steps, dtype=np.float32); self.rewards = np.zeros(self.n_steps, dtype=np.float32); self.dones = np.zeros(self.n_steps, dtype=np.float32); self.values = np.zeros(self.n_steps, dtype=np.float32); self.step = 0; self.computed = False
    def add(self, obs_t, action_t, log_prob_t, reward_t, done_t, value_t):
        if self.step >= self.n_steps: print("Warning: Rollout buffer overflow, overwriting."); self.step = 0
        self.obs[self.step] = obs_t; self.actions[self.step] = action_t; self.log_probs[self.step] = log_prob_t; self.rewards[self.step] = reward_t; self.dones[self.step] = float(done_t); self.values[self.step] = value_t; self.step += 1
    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        last_gae_lam = 0; num_valid_steps = self.step; self.advantages = np.zeros(num_valid_steps, dtype=np.float32); self.returns = np.zeros(num_valid_steps + 1, dtype=np.float32); self.returns[num_valid_steps] = last_value
        for t in reversed(range(num_valid_steps)):
            next_value = self.values[t + 1] if t < num_valid_steps - 1 else last_value
            next_non_terminal = 1.0 - self.dones[t + 1] if t < num_valid_steps - 1 else (1.0 - self.dones[t])
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]; last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam; self.advantages[t] = last_gae_lam
        self.returns = self.advantages + self.values[:num_valid_steps]; self.computed = True
    def get_batches(self, batch_size):
        if not self.computed: raise RuntimeError("Returns/advantages needed.")
        n_samples = self.step
        if n_samples == 0: return # Generator yields nothing if empty
        indices = np.arange(n_samples); np.random.shuffle(indices)
        valid_advantages = self.advantages[:n_samples]; norm_advantages = (valid_advantages - valid_advantages.mean()) / (valid_advantages.std() + 1e-8)
        obs_tensor = torch.tensor(self.obs[:n_samples], dtype=torch.float32).to(self.device); actions_tensor = torch.tensor(self.actions[:n_samples], dtype=torch.long).to(self.device)
        log_probs_tensor = torch.tensor(self.log_probs[:n_samples], dtype=torch.float32).to(self.device); values_tensor = torch.tensor(self.values[:n_samples], dtype=torch.float32).to(self.device)
        returns_tensor = torch.tensor(self.returns[:n_samples], dtype=torch.float32).to(self.device); advantages_tensor = torch.tensor(norm_advantages, dtype=torch.float32).to(self.device)
        start_idx = 0
        while start_idx < n_samples:
            batch_indices = indices[start_idx : start_idx + batch_size]
            yield (obs_tensor[batch_indices], actions_tensor[batch_indices], log_probs_tensor[batch_indices],
                   values_tensor[batch_indices], returns_tensor[batch_indices], advantages_tensor[batch_indices])
            start_idx += batch_size
    def after_update(self): self.reset()

# --- Running Stats ---
class RunningStats:
    # (Identical to the corrected RunningStats class defined previously)
    def __init__(self, shape=(), eps=1e-5): self.mean = np.zeros(shape, dtype=np.float64); self.var = np.ones(shape, dtype=np.float64); self.count = eps
    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 0: x = x[np.newaxis]
        if x.shape[0] == 0: return
        batch_mean, batch_var = np.mean(x, axis=0), np.var(x, axis=0); batch_count = x.shape[0]
        delta = batch_mean - self.mean; tot_count = self.count + batch_count; new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count; m_b = batch_var * batch_count; m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count; self.mean, self.var, self.count = new_mean, new_var, tot_count
    def normalize(self, x): x = np.asarray(x); std = np.sqrt(np.maximum(self.var, 1e-6)); return (x - self.mean) / std
    def get_state(self): return {'mean': self.mean, 'var': self.var, 'count': self.count}
    def set_state(self, state): self.mean, self.var, self.count = state['mean'], state['var'], state['count']

# --- Helper Functions ---
def get_driving_inputs(step, t_manager, w_manager, config):
    # (Identical to the previous PPO example)
    day, hour, time_features, manager_done = t_manager.step()
    ambient_temp, _, wet_bulb, _ = w_manager.step()
    load_profile_type=config.get('load_profile','variable')
    if load_profile_type=='sinusoidal': cpu_load=0.5+0.4*np.sin(2*np.pi*hour/24); gpu_load=0.2+0.15*np.sin(2*np.pi*day/7+np.pi/2); mem_util=0.5
    elif load_profile_type=='constant': cpu_load, gpu_load, mem_util=0.6, 0.2, 0.6
    else: cpu_load=np.clip(np.random.normal(0.5,0.2),0.1,0.9); gpu_load=np.clip(np.random.normal(0.2,0.1),0.0,0.5); mem_util=np.clip(np.random.normal(0.5,0.1),0.2,0.8)
    return cpu_load, gpu_load, mem_util, ambient_temp, wet_bulb, manager_done

def construct_hvac_observation(env_info, current_setpoint, cpu_load, gpu_load, current_hour, obs_dim):
    """ Constructs the observation vector based on desired features. MUST align with obs_dim. """
    # --- Critical: Extract features CONSISTENTLY with obs_dim ---
    try:
        # Get required values from info dict or parameters
        ambient_temp = env_info.get('dc_exterior_ambient_temp', 20.0) # Example default

        # Calculate time features
        sin_hour = np.sin(2 * np.pi * current_hour / 24.0)
        cos_hour = np.cos(2 * np.pi * current_hour / 24.0)
        
        # Normalize the current_setpoint
        # Assuming setpoint is in Celsius and we want to normalize it between 0 and 1
        min_temp, max_temp = 18.0, 27.0
        current_setpoint = (current_setpoint - min_temp) / (max_temp - min_temp)

        # Ensure order matches definition and obs_dim = 6
        obs_list = [
            sin_hour,
            cos_hour,
            ambient_temp,
            cpu_load,
            gpu_load,
            current_setpoint, # This is the *previous* absolute setpoint applied
        ]

        # --- Verification ---
        if len(obs_list) != obs_dim:
             # This should ideally not happen if obs_dim is set correctly based on this list
            raise ValueError(f"Observation dimension mismatch: expected {obs_dim}, got {len(obs_list)}")

        obs = np.array(obs_list, dtype=np.float32)
        return obs

    except KeyError as e:
        print(f"Error constructing observation: Missing key {e} in env_info dict.")
        return np.zeros(obs_dim, dtype=np.float32) # Return zero vector on error
    except Exception as e:
        print(f"Error constructing observation: {e}")
        return np.zeros(obs_dim, dtype=np.float32)

def calculate_hvac_reward(info, target_setpoint, min_sp=18.0, max_sp=27.0):
    # (Identical to the previous corrected PPO example)
    hvac_power_kw = info.get('dc_total_power_kW', 1000.0) 
    power_penalty = -(hvac_power_kw/100.0)
    boundary_penalty = -0.1 if target_setpoint<=min_sp+0.5 or target_setpoint>=max_sp-0.5 else 0
    temp_penalty = 0
    reward = power_penalty + boundary_penalty + temp_penalty
    return reward

# --- Main Training Function ---
def train_hvac_ppo():
    # --- Setup ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"HVAC_PPO_{timestamp}"
    log_dir = f"logs/{run_id}"; tb_dir = f"runs/{run_id}"; ckpt_dir = f"checkpoints/{run_id}"
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)
    logger = setup_logger(log_dir, enable_logger=True)

    # --- Load Configs ---
    dc_physics_json_path = "configs/dcs/dc_config.json"
    hvac_config_path = "configs/env/hvac_train_config_ppo.yaml"

    try:
        with open(dc_physics_json_path, 'r') as f: dc_physics_params_dict = json.load(f)
    except Exception as e: logger.error(f"Error loading physics config {dc_physics_json_path}: {e}"); return
    try:
        with open(hvac_config_path, 'r') as f: hvac_cfg = yaml.safe_load(f)['hvac_training']
    except Exception as e: logger.error(f"Error loading HVAC config {hvac_config_path}: {e}"); return

    # *** Use the actual DC_Config class ***
    try:
        example_cores = 50000
        example_gpus = 1000
        example_mem_gb = 80000
        example_dc_mw = 1.0
        dc_config_obj = DC_Config(
            dc_config_file=dc_physics_json_path, total_cores=example_cores,
            total_gpus=example_gpus, total_mem_GB=example_mem_gb,
            datacenter_capacity_mw=example_dc_mw )
        logger.info(f"Successfully loaded DC_Config from {dc_physics_json_path}")
    except Exception as e: logger.error(f"Error initializing DC_Config: {e}"); return

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {DEVICE}"); logger.info(f"Config: {hvac_cfg}")

    # --- Seeding ---
    seed = hvac_cfg.get('seed', 42); np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    # --- Environment ---
    hvac_obs_dim = hvac_cfg['hvac_obs_dim']
    action_space = spaces.Discrete(3)
    action_mapping = {0: -1.0, 1: 0.0, 2: 1.0}
    min_temp, max_temp = 18.0, 27.0
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(hvac_obs_dim,), dtype=np.float32)

    try:
        hvac_env = dc_gymenv(
            observation_variables=[], dc_memory_GB=example_mem_gb,
            observation_space=observation_space, action_variables=['Cooling_Setpoint_RL'],
            action_space=action_space, action_mapping=action_mapping, ranges={},
            add_cpu_usage=True, add_gpu_usage=True, min_temp=min_temp, max_temp=max_temp,
            action_definition={'cooling setpoints': {'name': 'Cooling_Setpoint_RL', 'initial_value': 22.0}},
            DC_Config=dc_config_obj, seed=seed )
        logger.info("dc_gymenv initialized successfully.")
    except Exception as e: logger.error(f"Error initializing dc_gymenv: {e}"); return

    # --- Agent ---
    actor = ActorNet(hvac_obs_dim, action_space.n, hvac_cfg['hidden_dim']).to(DEVICE)
    critic = ValueNet(hvac_obs_dim, hvac_cfg['hidden_dim']).to(DEVICE)
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=float(hvac_cfg['learning_rate']), eps=1e-5)

    # --- Storage & Normalization ---
    storage = PPORolloutStorage(hvac_cfg['n_steps'], hvac_obs_dim, DEVICE)
    obs_stats = RunningStats(shape=(hvac_obs_dim,))

    # --- Driving Data Managers ---
    start_day=random.randint(30*6, 30*7) # Start in July
    start_hour=random.randint(0, 23)
    sim_duration_days = hvac_cfg.get('simulation_duration_days', 7) # Example default
    t_m = Time_Manager(init_day=start_day,
                        timezone_shift=0, # Usually 0 for training env
                        duration_days=sim_duration_days)
    weather_m = Weather_Manager(location=hvac_cfg['location'], simulation_year=hvac_cfg['simulation_year'], timezone_shift=0)
    t_m.reset(init_day=start_day, init_hour=start_hour, seed=seed)
    weather_m.reset(init_day=start_day, init_hour=start_hour, seed=seed)
    current_hour = start_hour # <<<--- TRACK CURRENT HOUR

    # --- Training Loop ---
    logger.info("Starting HVAC PPO Training...")
    current_setpoint = 22.0

    try: # Wrap main loop for better error reporting during init obs
        _, current_info = hvac_env.reset(seed=seed)
        # Get driving inputs for the *first* step (step 0)
        # Need hour from t_m state *after* reset
        current_day_init, current_hour_init, _ = t_m.day, t_m.hour, t_m.step() # Use internal state
        t_m.reset(init_day=start_day, init_hour=start_hour, seed=seed) # Reset again to be at step 0
        cpu_load, gpu_load, mem_util, ambient_temp, wet_bulb, manager_done = get_driving_inputs(0, t_m, weather_m, hvac_cfg)

        hvac_env.set_ambient_temp(ambient_temp, wet_bulb); hvac_env.update_workloads(cpu_load, gpu_load, mem_util)
        # Construct initial observation based on initial info, state and hour
        obs_agent_raw = construct_hvac_observation(
            current_info, current_setpoint, cpu_load, gpu_load, current_hour_init, hvac_obs_dim # Pass initial hour
        )
        obs_stats.update(obs_agent_raw); obs_agent_norm = obs_stats.normalize(obs_agent_raw)
    except Exception as e: logger.error(f"Error during initial reset/observation: {e}"); return
    
    global_step = 0
    num_updates = hvac_cfg['total_steps'] // hvac_cfg['n_steps']

    # --- Episode Metric Tracking ---
    episode_reward = 0
    episode_steps = 0
    episode_rewards_list = deque(maxlen=10) # For MeanEpReward_Last10
    episode_hvac_powers = []; episode_ite_powers = []; episode_total_powers = []
    episode_return_temps = []; episode_setpoints = []
    # --- End episode metric tracking ---

    for update in tqdm(range(num_updates), desc="PPO Updates"):
        actor.eval(); critic.eval()
        rewards_this_rollout = []
        # *** Clear rollout metrics ***
        rollout_hvac_power = []; rollout_ite_power = []; rollout_total_power = []
        rollout_return_temps = []; rollout_setpoints = []
        # *** End clear metrics ***
        # *** End clear metrics ***


        # --- Rollout Phase ---
        for step in range(hvac_cfg['n_steps']):
            global_step += 1
            try:
                # Sample Action
                obs_tensor = torch.FloatTensor(obs_agent_norm).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    action_logits=actor(obs_tensor); dist=torch.distributions.Categorical(logits=action_logits)
                    action_discrete=dist.sample(); log_prob=dist.log_prob(action_discrete); value=critic(obs_tensor)
                action_discrete_np=action_discrete.cpu().numpy().flatten()[0]

                # Translate Action & Step Env
                setpoint_delta=action_mapping[action_discrete_np]
                target_setpoint=np.clip(current_setpoint+setpoint_delta, min_temp, max_temp)

                # Get driving inputs for *next* state
                next_cpu_load, next_gpu_load, next_mem_util, next_ambient_temp, next_wet_bulb, manager_done = get_driving_inputs(
                    global_step, t_m, weather_m, hvac_cfg)
                next_hour = t_m.hour # Get hour *after* manager step

                hvac_env.set_ambient_temp(next_ambient_temp, next_wet_bulb); hvac_env.update_workloads(next_cpu_load, next_gpu_load, next_mem_util)
                _, _, terminated, truncated, next_info = hvac_env.step(target_setpoint)
                done = terminated or truncated or manager_done

                reward = calculate_hvac_reward(next_info, target_setpoint, min_temp, max_temp)
                rewards_this_rollout.append(reward)
                episode_reward += reward # Accumulate for episode log
                episode_steps += 1

                # *** Store step metrics for rollout average calculation ***
                rollout_hvac_power.append(next_info.get('dc_HVAC_total_power_kW', 0)) # Use corrected key
                rollout_ite_power.append(next_info.get('dc_ITE_total_power_kW', 0))
                rollout_total_power.append(next_info.get('dc_total_power_kW', 0))
                rollout_return_temps.append(next_info.get('dc_avg_return_temperature', 25.0))
                rollout_setpoints.append(target_setpoint)
                # Also store for episode average
                episode_hvac_powers.append(next_info.get('dc_HVAC_total_power_kW', 0))
                episode_ite_powers.append(next_info.get('dc_ITE_total_power_kW', 0))
                episode_total_powers.append(next_info.get('dc_total_power_kW', 0))
                episode_return_temps.append(next_info.get('dc_avg_return_temperature', 25.0))
                episode_setpoints.append(target_setpoint)
                 # *** End storing step metrics ***

                storage.add(obs_agent_norm, action_discrete_np, log_prob.item(), reward, done, value.item())

                # Prepare observation for the *next* iteration
                prev_setpoint_for_next_obs = current_setpoint # The setpoint that *led* to next_info
                current_setpoint = target_setpoint # Update tracked setpoint *after* step for the next action decision
                
                obs_next_raw = construct_hvac_observation(
                    next_info,                  # Info corresponding to the state we landed in
                    prev_setpoint_for_next_obs, # The setpoint that was active *during* this step
                    next_cpu_load,              # Load active *during* this step
                    next_gpu_load,              # Load active *during* this step
                    next_hour,                  # Hour corresponding to the end of this step / start of next
                    hvac_obs_dim
                )
                obs_stats.update(obs_next_raw)
                obs_agent_norm = obs_stats.normalize(obs_next_raw)

                # Handle episode termination
                if done:
                    # *** Log Episode Metrics ***
                    episode_rewards_list.append(episode_reward)
                    writer.add_scalar("Episode/TotalReward", episode_reward, global_step)
                    writer.add_scalar("Episode/Length", episode_steps, global_step)
                    if len(episode_rewards_list) >= 10:
                         writer.add_scalar("Perf/MeanEpReward_Last10", np.mean(episode_rewards_list), global_step)
                    # Log average metrics for the completed episode
                    if episode_steps > 0:
                        writer.add_scalar("Episode/AvgHVACPower_kW", np.mean(episode_hvac_powers), global_step)
                        writer.add_scalar("Episode/AvgITEPower_kW", np.mean(episode_ite_powers), global_step)
                        writer.add_scalar("Episode/AvgTotalPower_kW", np.mean(episode_total_powers), global_step)
                        writer.add_scalar("Episode/AvgReturnTemp_C", np.mean(episode_return_temps), global_step)
                        writer.add_scalar("Episode/AvgSetpoint_C", np.mean(episode_setpoints), global_step)
                    # *** End Episode Logging ***

                    # --- Reset for next episode ---
                    current_setpoint = 22.0; _, current_info = hvac_env.reset(seed=seed + global_step)
                    start_day_july=random.randint(182,212); start_hour_july=random.randint(0,23)
                    t_m.reset(init_day=start_day_july, init_hour=start_hour_july, seed=seed + global_step)
                    weather_m.reset(init_day=start_day_july, init_hour=start_hour_july, seed=seed + global_step)
                    cpu_load, gpu_load, mem_util, ambient_temp, wet_bulb, _ = get_driving_inputs(global_step, t_m, weather_m, hvac_cfg)
                    current_hour = t_m.hour
                    hvac_env.set_ambient_temp(ambient_temp, wet_bulb); hvac_env.update_workloads(cpu_load, gpu_load, mem_util)
                    obs_agent_raw = construct_hvac_observation(current_info, current_setpoint, cpu_load, gpu_load, current_hour, hvac_obs_dim)
                    obs_agent_norm = obs_stats.normalize(obs_agent_raw)
                    # Reset episode trackers
                    episode_reward = 0; episode_steps = 0;
                    episode_hvac_powers.clear(); episode_ite_powers.clear(); episode_total_powers.clear()
                    episode_return_temps.clear(); episode_setpoints.clear()
                    # --- End Reset ---
                else:
                    # Update current_hour if not done
                    current_hour = next_hour

            except Exception as e: logger.error(f"Error during rollout step {global_step}: {e}"); break

        # --- Compute Returns & Advantages ---
        try:
            with torch.no_grad():
                last_obs_tensor = torch.FloatTensor(obs_agent_norm).unsqueeze(0).to(DEVICE); last_value = critic(last_obs_tensor).item()
            storage.compute_returns_and_advantages(last_value, hvac_cfg['gamma'], hvac_cfg['gae_lambda'])
        except Exception as e: logger.error(f"Error computing returns/advantages: {e}"); storage.after_update(); continue

        # --- Update Phase ---
        actor.train(); critic.train()
        policy_loss_epoch, value_loss_epoch, entropy_epoch = 0, 0, 0
        approx_kl_epoch = 0 # Track KL divergence
        clip_frac_epoch = 0 # Track fraction of clipped ratios
        num_minibatches = 0

        try:
            for epoch in range(hvac_cfg['num_epochs']):
                for batch_data in storage.get_batches(hvac_cfg['batch_size']):
                    if batch_data is None: continue
                    obs_b, actions_b, old_log_probs_b, _, returns_b, advantages_b = batch_data
                    if len(obs_b) == 0: continue
                    num_minibatches +=1

                    # Evaluate current policy
                    action_logits=actor(obs_b); dist=torch.distributions.Categorical(logits=action_logits)
                    new_log_probs=dist.log_prob(actions_b); entropy=dist.entropy().mean(); new_values=critic(obs_b).squeeze(-1)

                    # Policy Loss
                    log_ratio=new_log_probs-old_log_probs_b; ratio=torch.exp(log_ratio)
                    surr1=ratio*advantages_b; surr2=torch.clamp(ratio,1.0-hvac_cfg['clip_coef'],1.0+hvac_cfg['clip_coef'])*advantages_b
                    policy_loss=-torch.min(surr1,surr2).mean()
                    # Value Loss
                    value_loss=F.mse_loss(new_values,returns_b)
                    # Total Loss
                    loss=policy_loss-hvac_cfg['ent_coef']*entropy+hvac_cfg['vf_coef']*value_loss

                    # --- Calculate PPO-specific diagnostics ---
                    with torch.no_grad():
                         approx_kl = ((ratio - 1) - log_ratio).mean().item() # Approximate KL divergence
                         clipped = ratio.gt(1 + hvac_cfg['clip_coef']) | ratio.lt(1 - hvac_cfg['clip_coef'])
                         clip_fraction = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
                    approx_kl_epoch += approx_kl
                    clip_frac_epoch += clip_fraction
                    # --- End diagnostics ---

                    # Optimization
                    optimizer.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(actor.parameters())+list(critic.parameters()),hvac_cfg['max_grad_norm'])
                    optimizer.step()

                    policy_loss_epoch+=policy_loss.item(); value_loss_epoch+=value_loss.item(); entropy_epoch+=entropy.item()
        except Exception as e: logger.error(f"Error during PPO update phase: {e}")


        # --- Logging ---
        if num_minibatches > 0:
            avg_policy_loss=policy_loss_epoch/num_minibatches; avg_value_loss=value_loss_epoch/num_minibatches; avg_entropy=entropy_epoch/num_minibatches
            avg_approx_kl = approx_kl_epoch / num_minibatches
            avg_clip_frac = clip_frac_epoch / num_minibatches
            mean_rollout_reward = np.mean(rewards_this_rollout) if rewards_this_rollout else 0

            # *** Log additional metrics to TensorBoard ***
            writer.add_scalar("Perf/MeanRolloutReward", mean_rollout_reward, global_step)
            writer.add_scalar("Loss/PolicyLoss", avg_policy_loss, global_step)
            writer.add_scalar("Loss/ValueLoss", avg_value_loss, global_step)
            writer.add_scalar("Loss/Entropy", avg_entropy, global_step)
            writer.add_scalar("PPO/ApproxKL", avg_approx_kl, global_step)
            writer.add_scalar("PPO/ClipFraction", avg_clip_frac, global_step)
            writer.add_scalar("Perf/MeanHVACPower_kW", np.mean(rollout_hvac_power) if rollout_hvac_power else 0, global_step)
            writer.add_scalar("Perf/MeanITEPower_kW", np.mean(rollout_ite_power) if rollout_ite_power else 0, global_step)
            writer.add_scalar("Perf/MeanTotalPower_kW", np.mean(rollout_total_power) if rollout_total_power else 0, global_step)
            writer.add_scalar("Perf/MeanReturnTemp_C", np.mean(rollout_return_temps) if rollout_return_temps else 0, global_step)
            writer.add_scalar("Perf/MeanSetpoint_C", np.mean(rollout_setpoints) if rollout_setpoints else 0, global_step)
            # *** End additional logging ***

            if update % (hvac_cfg['log_interval'] // hvac_cfg['n_steps'] + 1) == 0:
                log_msg = (f"[Update {update+1}/{num_updates}, Step {global_step}] "
                           f"Rew: {mean_rollout_reward:.3f}, "
                           f"P Loss: {avg_policy_loss:.3f}, V Loss: {avg_value_loss:.3f}, "
                           f"Entropy: {avg_entropy:.3f}, KL: {avg_approx_kl:.3f}, ClipFrac: {avg_clip_frac:.3f}")
                logger.info(log_msg)
                tqdm.write(log_msg) # Use tqdm.write inside loop


        # --- Clear Storage ---
        storage.after_update()

        # --- Checkpointing ---
        if update > 0 and update % (hvac_cfg['save_interval'] // hvac_cfg['n_steps'] + 1) == 0 :
            try:
                save_data = {'actor_state_dict': actor.state_dict(), 'hvac_obs_dim': hvac_obs_dim,
                             'obs_stats_state': obs_stats.get_state() if obs_stats else None, 'global_step': global_step }
                ckpt_path = os.path.join(ckpt_dir, f"hvac_ppo_{global_step}.pth")
                torch.save(save_data, ckpt_path)
                logger.info(f"Saved HVAC checkpoint to {ckpt_path}")
            except Exception as e: logger.error(f"Error saving checkpoint at step {global_step}: {e}")

    # --- Final Save ---
    try:
        final_save_path = hvac_cfg['policy_save_path']
        final_save_data = {'actor_state_dict': actor.state_dict(), 'hvac_obs_dim': hvac_obs_dim,
                           'obs_stats_state': obs_stats.get_state() if obs_stats else None,}
        torch.save(final_save_data, final_save_path)
        logger.info(f"Training finished. Final HVAC policy saved to {final_save_path}")
    except Exception as e: logger.error(f"Error saving final policy: {e}")
    writer.close()



if __name__ == "__main__":
    # Example: Ensure configs exist before running
    if not os.path.exists("configs/env/hvac_train_config_ppo.yaml"):
        print("Error: Please create 'configs/env/hvac_train_config_ppo.yaml'")
    elif not os.path.exists("configs/dcs/dc_config.json"):
         print("Error: Please ensure 'configs/dcs/dc_config.json' exists")
    else:
        train_hvac_ppo()