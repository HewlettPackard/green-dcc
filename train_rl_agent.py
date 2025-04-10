import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from tqdm import trange

from envs.task_scheduling_env import TaskSchedulingEnv
from rl_components.agent_net import ActorNet, CriticNet
from rl_components.replay_buffer import ReplayBuffer, FastReplayBuffer
from rl_components.replay_buffer import PrioritizedReplayBuffer
from rewards.predefined.energy_price_reward import EnergyPriceReward
from rewards.predefined.composite_reward import CompositeReward

import logging
import os
import datetime

from torch.utils.tensorboard import SummaryWriter

from collections import deque

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

CHECKPOINT_DIR = f"checkpoints/train_{timestamp}"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(step, actor, critic, actor_opt, critic_opt, best=False):
    checkpoint = {
        "step": step,
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "actor_optimizer_state_dict": actor_opt.state_dict(),
        "critic_optimizer_state_dict": critic_opt.state_dict(),
    }
    print(f"Saving checkpoint at step {step} to {CHECKPOINT_DIR}")
    if best:
        torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "best_checkpoint.pth"))
    else:
        torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f"checkpoint_step_{step}.pth"))

def load_checkpoint(path, actor, critic, actor_opt=None, critic_opt=None):
    checkpoint = torch.load(path, map_location=DEVICE)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    critic.load_state_dict(checkpoint["critic_state_dict"])
    if actor_opt and critic_opt:
        actor_opt.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        critic_opt.load_state_dict(checkpoint["critic_optimizer_state_dict"])
    return checkpoint.get("step", 0)


# === Set up TensorBoard ===
writer = SummaryWriter(log_dir=f"runs/train_{timestamp}")

# === Set up logger ===
os.makedirs("logs", exist_ok=True)
log_path = f"logs/train_{timestamp}.log"

# === Root logger
logger = None#
# logger = logging.getLogger("train_logger")
# logger.setLevel(logging.INFO)  # File handler will capture INFO+

# # === File handler (full log)
# file_handler = logging.FileHandler(log_path, mode="w")
# file_handler.setLevel(logging.INFO)
# file_handler.setFormatter(logging.Formatter(
#     "%(asctime)s - %(levelname)s - %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S"
# ))
# logger.addHandler(file_handler)

# # === Console handler (only warnings and errors)
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.WARNING)  # Only show warnings+errors in terminal
# console_handler.setFormatter(logging.Formatter(
#     "[%(levelname)s] %(message)s"
# ))
# logger.addHandler(console_handler)

# === CONFIG ===
GAMMA = 0.99
ALPHA = 0.01
LR = 1e-4
BATCH_SIZE = 512
TAU = 0.005
REPLAY_SIZE = int(1e5)
WARMUP_STEPS = 1000
TOTAL_STEPS = int(1e7)
UPDATE_FREQ = 1
POLICY_UPDATE_FREQ = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_INTERVAL = 5000  # Save every 5k steps

print(f"Using device: {DEVICE}")


# === INIT ENV (Replace this with your real cluster manager setup) ===
def make_env():
    import pandas as pd
    import datetime
    from simulation.cluster_manager import DatacenterClusterManager
    from envs.task_scheduling_env import TaskSchedulingEnv

    # === Simulation time range ===
    simulation_year = 2023
    simulated_month = 8
    init_day_month = 1
    init_hour = 5
    init_minute = 0

    start_time = datetime.datetime(simulation_year, simulated_month, init_day_month, init_hour, init_minute, tzinfo=datetime.timezone.utc)
    end_time = start_time + datetime.timedelta(days=7)
    start_time = pd.Timestamp(start_time)
    end_time = pd.Timestamp(end_time)

    # === Datacenter configurations ===
    datacenter_configs = [
        {
            'location': 'US-NY-NYIS', 'dc_id': 1, 'agents': [], 'timezone_shift': -5,
            'dc_config_file': 'dc_config.json', 'weather_file': None, 'cintensity_file': None,
            'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv', 'month': simulated_month,
            'datacenter_capacity_mw': 1.5, 'max_bat_cap_Mw': 3.0, 'days_per_episode': 30,
            'network_cost_per_gb': 0.08, 'total_cpus': 5000, 'total_gpus': 700,
            'total_mem': 5000, 'population_weight': 0.25,
        },
        {
            'location': 'DE-LU', 'dc_id': 2, 'agents': [], 'timezone_shift': 1,
            'dc_config_file': 'dc_config.json', 'weather_file': None, 'cintensity_file': None,
            'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv', 'month': simulated_month,
            'datacenter_capacity_mw': 1.2, 'max_bat_cap_Mw': 2.5, 'days_per_episode': 30,
            'network_cost_per_gb': 0.07, 'total_cpus': 5000, 'total_gpus': 700,
            'total_mem': 5000, 'population_weight': 0.22,
        },
        {
            'location': 'ZA', 'dc_id': 3, 'agents': [], 'timezone_shift': 2,
            'dc_config_file': 'dc_config.json', 'weather_file': None, 'cintensity_file': None,
            'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv', 'month': simulated_month,
            'datacenter_capacity_mw': 1.0, 'max_bat_cap_Mw': 2.0, 'days_per_episode': 30,
            'network_cost_per_gb': 0.06, 'total_cpus': 5000, 'total_gpus': 700,
            'total_mem': 5000, 'population_weight': 0.13,
        },
        {
            'location': 'SG', 'dc_id': 4, 'agents': [], 'timezone_shift': 8,
            'dc_config_file': 'dc_config.json', 'weather_file': None, 'cintensity_file': None,
            'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv', 'month': simulated_month,
            'datacenter_capacity_mw': 1.8, 'max_bat_cap_Mw': 3.5, 'days_per_episode': 30,
            'network_cost_per_gb': 0.09, 'total_cpus': 5000, 'total_gpus': 700,
            'total_mem': 5000, 'population_weight': 0.25,
        },
        {
            'location': 'AU-NSW', 'dc_id': 5, 'agents': [], 'timezone_shift': 11,
            'dc_config_file': 'dc_config.json', 'weather_file': None, 'cintensity_file': None,
            'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv', 'month': simulated_month,
            'datacenter_capacity_mw': 1.4, 'max_bat_cap_Mw': 2.8, 'days_per_episode': 30,
            'network_cost_per_gb': 0.10, 'total_cpus': 5000, 'total_gpus': 700,
            'total_mem': 5000, 'population_weight': 0.15,
        }
    ]

    # === Workload data ===
    tasks_file_path = "data/workload/alibaba_2020_dataset/result_df_full_year_2020.pkl"
    
    # === Create cluster manager ===
    cluster_manager = DatacenterClusterManager(
        config_list=datacenter_configs,
        simulation_year=simulation_year,
        init_day=int(simulated_month*30.5),
        init_hour=init_hour,
        strategy="manual_rl",
        tasks_file_path=tasks_file_path,
        shuffle_datacenter_order=False,  # shuffle only during training
        cloud_provider='gcp',
        logger=logger,
    )
    
    # cluster_manager.logger = logger

    # === Wrap into Gym environment ===
    # reward_fn = EnergyPriceReward(normalize_factor=100000)
    reward_fn = CompositeReward(
        components={
            "energy_price": {
                "weight": 0.5,
                "args": {"normalize_factor": 100000}
            },
            "carbon_emissions": {
                "weight": 0.3,
                "args": {"normalize_factor": 10}
            },
            # "sla_penalty": {
            #     "weight": 0.2,
            #     "args": {"penalty_per_violation": 5.0}
            # }
            "transmission_cost": {
                "weight": 0.3,
                "args": {"normalize_factor": 1}
            },
        },
        normalize=False
    )

    env = TaskSchedulingEnv(
        cluster_manager=cluster_manager,
        start_time=start_time,
        end_time=end_time,
        reward_fn=reward_fn,
        writer=writer,
    )

    return env

class RunningStats:
    def __init__(self, eps=1e-5):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps

    def update(self, x):
        x = float(x)
        self.count += 1
        last_mean = self.mean
        self.mean += (x - self.mean) / self.count
        self.var += (x - last_mean) * (x - self.mean)

    def normalize(self, x):
        std = max(np.sqrt(self.var / self.count), 1e-6)
        return (x - self.mean) / std


############################
# Main training loop
############################
def train():
    # 1) Create environment
    env = make_env()
    reward_stats = RunningStats()
    episode_reward_buffer = deque(maxlen=10)

    # 2) Dynamically detect obs_dim
    # We'll do a dummy reset to get the shape from the first obs.
    # obs is a list of shape (N, obs_dim). If no tasks, we retry steps.
    obs, _ = env.reset()
    while len(obs) == 0:
        # If zero tasks, step the env with an empty action
        # Or do something minimal.
        # We'll just step forward with an empty list.
        next_obs, _, done, _, _ = env.step([])
        obs = next_obs
        if done:
            obs, _ = env.reset()

    obs_dim = len(obs[0])  # Number of features per task
    max_tasks = 500
    if logger:
        logger.info(f"Detected obs_dim={obs_dim}. Using max_tasks={max_tasks}.")
    # logger.info(f"Detected obs_dim={obs_dim}. Using max_tasks={max_tasks}.")

    # 3) Create actor & critic networks
    hidden_dim = 64
    # For the action: 
        # Each element ∈ [0, num_dcs]
            # where 0 = "defer task"
            # and [1, num_dcs] = assign to DC (1-based index)
    act_dim = env.num_dcs + 1

    actor = ActorNet(obs_dim, act_dim, hidden_dim=hidden_dim).to(DEVICE)
    critic = CriticNet(obs_dim, act_dim, hidden_dim=hidden_dim).to(DEVICE)
    target_critic = CriticNet(obs_dim, act_dim, hidden_dim=hidden_dim).to(DEVICE)

    target_critic.load_state_dict(critic.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=LR)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=LR)

    # 4) Create replay buffer
    # buffer = ReplayBuffer(
    #     capacity=REPLAY_SIZE,
    #     max_tasks=max_tasks,
    #     obs_dim=obs_dim
    # )
    buffer = FastReplayBuffer(
        capacity=REPLAY_SIZE,
        max_tasks=max_tasks,
        obs_dim=obs_dim
    )
    # buffer = PrioritizedReplayBuffer(
        #     capacity=REPLAY_SIZE,
        #     max_tasks=max_tasks,
        #     obs_dim=obs_dim,
        #     alpha=0.6  # Level of prioritization
        # )

    # Since we already called env.reset above, we can proceed.
    log_interval = 100
    episode_reward = 0
    episode_steps = 0
    best_reward = float("-inf")

    # We'll define a variable for the step:
    global_step = 0
    
    q_loss = None
    policy_loss = None
    pbar = trange(TOTAL_STEPS)

    for global_step in pbar:
        # We have obs => shape (N, obs_dim)
        if len(obs) == 0:
            # no tasks => do an empty action list.
            actions = []
            if global_step % log_interval == 0:
                writer.add_scalar("Meta/EmptyObs", int(len(obs) == 0), global_step)
        else:
            # 5) Choose actions
            obs_tensor = torch.FloatTensor(obs).to(DEVICE)
            if global_step < WARMUP_STEPS:
                actions = [np.random.randint(env.num_dcs + 1) for _ in obs]
            else:
                with torch.no_grad():
                    logits = actor(obs_tensor)  # shape (N, act_dim)
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    actions = dist.sample().cpu().numpy().tolist()
                    
                    assert all(0 <= a < act_dim for a in actions), f"Sampled invalid action: {actions}"

                    
                writer.add_histogram("Actor/logits", logits, global_step)
                writer.add_histogram("Actor/probs", probs, global_step)
        
        # 6) Step environment
        next_obs, reward, done, truncated, _ = env.step(actions)
        reward_stats.update(reward)
        normalized_reward = reward_stats.normalize(reward)

        # === Skip buffer update if no real tasks (i.e., no actions taken)
        if len(actions) == 0:
            obs = next_obs
            continue

        done_flag = done or truncated
        buffer.add(obs, actions, normalized_reward, next_obs, done_flag)

        obs = next_obs
        episode_reward += reward
        episode_steps += 1
        if done_flag:
            avg_reward = episode_reward / episode_steps
            episode_reward_buffer.append(avg_reward)

            if logger:
                logger.info(f"[Episode End] total_reward={avg_reward:.2f}")
            pbar.write(f"[Episode End] total_reward={avg_reward:.2f}")
            writer.add_scalar("Reward/Episode_Reward", avg_reward, global_step)

            
            if len(episode_reward_buffer) == 10:
                rolling_avg_reward = sum(episode_reward_buffer) / 10
                writer.add_scalar("Reward/Avg10", rolling_avg_reward, global_step)

                if rolling_avg_reward > best_reward and global_step > 10 * WARMUP_STEPS:
                    best_reward = rolling_avg_reward
                    save_checkpoint(global_step, actor, critic, actor_opt, critic_opt, best=True)
                    pbar.write(f"[BEST] Saved checkpoint at step {global_step} (avg10 reward={rolling_avg_reward:.2f})")



            obs, _ = env.reset(seed=global_step)
            episode_reward = 0
            episode_steps = 0

        # 7) RL updates
        if global_step >= WARMUP_STEPS and global_step % UPDATE_FREQ == 0:
            if len(buffer) < BATCH_SIZE:
                if global_step % log_interval == 0:
                    print(f"[Buffer] Skipping update at step {global_step}. Buffer has only {len(buffer)} samples.")
                continue  # Skip update if not enough data
            # sample from buffer
            (obs_b, act_b, rew_b, next_obs_b, done_b,
             mask_obs_b, mask_next_b) = buffer.sample(BATCH_SIZE)

            obs_b = obs_b.to(DEVICE)         # [B, max_tasks, obs_dim]
            act_b = act_b.to(DEVICE)         # [B, max_tasks]
            rew_b = rew_b.to(DEVICE)         # [B]
            next_obs_b = next_obs_b.to(DEVICE)
            done_b = done_b.to(DEVICE)       # [B]
            mask_obs_b = mask_obs_b.to(DEVICE)
            mask_next_b = mask_next_b.to(DEVICE)

            B, T, D = obs_b.shape

            # Flatten current obs
            obs_flat = obs_b.view(B*T, D)
            act_flat = act_b.view(B*T)
            mask_obs_flat = mask_obs_b.view(B*T)
            
            masked_actions = act_flat[mask_obs_flat.bool()]
            if (masked_actions > act_dim).any() or (masked_actions < 0).any():
                print("Invalid action index detected:", masked_actions.min().item(), masked_actions.max().item())

            # Flatten next obs
            next_flat = next_obs_b.view(B*T, D)
            mask_next_flat = mask_next_b.view(B*T)

            with torch.no_grad():
                next_logits = actor(next_flat)
                next_probs = F.softmax(next_logits, dim=-1)
                next_log_probs = F.log_softmax(next_logits, dim=-1)
                q1_next, q2_next = target_critic.forward_all(next_flat)
                q_next = torch.min(q1_next, q2_next)
                # v_next shape: (B*T,)
                v_next = (next_probs * (q_next - ALPHA * next_log_probs)).sum(dim=-1)
                v_next = v_next * mask_next_flat
                # reshape to (B, T)
                v_next = v_next.view(B, T)

                # q_target shape: (B, T)
                q_target = rew_b.unsqueeze(1) + GAMMA * (1 - done_b.unsqueeze(1)) * v_next

            # compute Q values for actual (obs, action) pairs
            valid_idx = (act_flat >= 0) & (act_flat < act_dim)
            obs_valid = obs_flat[valid_idx]
            act_valid = act_flat[valid_idx]

            q1_all, q2_all = critic(obs_valid, act_valid)  # shape: [valid_tasks]
            # q1_all, q2_all = critic(obs_flat, act_flat)  # [B*T], [B*T]

            # mask invalid tasks
            q1_all_full = torch.zeros_like(act_flat, dtype=torch.float, device=DEVICE)
            q2_all_full = torch.zeros_like(act_flat, dtype=torch.float, device=DEVICE)
            q1_all_full[valid_idx] = q1_all
            q2_all_full[valid_idx] = q2_all

            q1_all = q1_all_full * mask_obs_flat
            q2_all = q2_all_full * mask_obs_flat

            # Reshape to [B, T]
            q1_chosen = q1_all.view(B, T)
            q2_chosen = q2_all.view(B, T)

            # compute MSE loss with target
            q_target = q_target * mask_obs_b

            q1_flat = q1_chosen.view(-1)
            q2_flat = q2_chosen.view(-1)
            target_flat = q_target.view(-1)
            mask_flat = mask_obs_b.view(-1).bool()

            q1_loss = F.mse_loss(q1_flat[mask_flat], target_flat[mask_flat], reduction='mean')
            q2_loss = F.mse_loss(q2_flat[mask_flat], target_flat[mask_flat], reduction='mean')
            q_loss = 0.5 * (q1_loss + q2_loss)

            critic_opt.zero_grad()
            q_loss.backward()
            critic_opt.step()

            # === Update policy
            if global_step % POLICY_UPDATE_FREQ == 0:
                logits = actor(obs_flat)
                probs = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)
                q1_eval, q2_eval = critic.forward_all(obs_flat)
                q_eval = torch.min(q1_eval, q2_eval)

                # shape of probs, q_eval, log_probs => (B*T, act_dim)
                # compute the policy loss with mask
                policy_loss = (probs * (ALPHA * log_probs - q_eval)).sum(dim=-1)
                policy_loss = policy_loss * mask_obs_flat
                policy_loss = policy_loss.sum() / mask_obs_flat.sum()
                
                entropy = torch.distributions.Categorical(probs).entropy()
                entropy = (entropy * mask_obs_flat).sum() / mask_obs_flat.sum()
                writer.add_scalar("Actor/Entropy", entropy.item(), global_step)

                actor_opt.zero_grad()
                policy_loss.backward()
                actor_opt.step()

                # soft update
                for p, tp in zip(critic.parameters(), target_critic.parameters()):
                    tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

        if global_step % log_interval == 0 and global_step > 0:
            if q_loss is not None and policy_loss is not None:
                writer.add_scalar("Loss/Q_Loss", q_loss.item(), global_step)
                writer.add_scalar("Loss/Policy_Loss", policy_loss.item(), global_step)

                if logger:
                    logger.info(f"[Step {global_step}] Q Loss: {q_loss.item():.3f} | Policy Loss: {policy_loss.item():.3f}")
                pbar.write(f"[Step {global_step}] Q Loss: {q_loss.item():.3f} | Policy Loss: {policy_loss.item():.3f}")
            else:
                if logger:
                    logger.info(f"[Step {global_step}] Skipping logging — losses not available yet.")
        

        if global_step % SAVE_INTERVAL == 0 and global_step > 0:
            save_checkpoint(global_step, actor, critic, actor_opt, critic_opt)
            pbar.write(f"Saved checkpoint at step {global_step}")

    writer.close()

if __name__ == "__main__":
    train()
