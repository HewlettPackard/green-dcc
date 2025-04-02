import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from tqdm import trange

from dc_env.dc_scheduling_env import TaskSchedulingEnv
from dc_env.agent_net import ActorNet, CriticNet
from dc_env.replay_buffer import ReplayBuffer

import logging
import os
import datetime

from torch.utils.tensorboard import SummaryWriter

import multiprocessing as mp
import random

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

CHECKPOINT_DIR = f"checkpoints/train_{timestamp}"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(step, actor, critic, actor_opt, critic_opt):
    checkpoint = {
        "step": step,
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "actor_optimizer_state_dict": actor_opt.state_dict(),
        "critic_optimizer_state_dict": critic_opt.state_dict(),
    }
    print(f"Saving checkpoint at step {step} to {CHECKPOINT_DIR}")
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
logger = None#logging.getLogger("train_logger")
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
NUM_WORKERS = 20
ROLLOUT_LEN = 1

GAMMA = 0.99
ALPHA = 0.01
LR = 1e-4
BATCH_SIZE = 256
TAU = 0.005
REPLAY_SIZE = 1_000_000
WARMUP_STEPS = 1_000
TOTAL_STEPS = 100_000
UPDATE_FREQ = 1
POLICY_UPDATE_FREQ = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_INTERVAL = 5000  # Save every 5k steps
TRANSITIONS_PER_ITER = NUM_WORKERS * ROLLOUT_LEN

print(f"Using device: {DEVICE}")


# === INIT ENV (Replace this with your real cluster manager setup) ===
def make_env(seed_offset=0):
    import pandas as pd
    import datetime
    from simulation.datacenter_cluster_manager import DatacenterClusterManager
    from dc_env.dc_scheduling_env import TaskSchedulingEnv

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
        tasks_file_path=tasks_file_path
    )
    
    cluster_manager.logger = logger

    # === Wrap into Gym environment ===
    env = TaskSchedulingEnv(
        cluster_manager=cluster_manager,
        start_time=start_time,
        end_time=end_time,
        carbon_price_per_kg=0.1  # tweak if needed
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


def rollout_worker(worker_id, queue):
    env = make_env(seed_offset=worker_id)
    obs, _ = env.reset(seed=42 + worker_id)
    buffer = []

    while True:
        if len(obs) == 0:
            actions = []
        else:
            actions = [random.randint(0, env.num_dcs - 1) for _ in obs]

        next_obs, reward, done, _, _ = env.step(actions)

        transition = (obs, actions, reward, next_obs, done)
        buffer.append(transition)

        obs = next_obs if not done else env.reset(seed=42 + worker_id)[0]

        if len(buffer) >= ROLLOUT_LEN:
            queue.put(buffer)
            buffer = []

############################
# Main training loop
############################
def train():
    # Start experience queue and rollout workers
    queue = mp.Queue()
    processes = [mp.Process(target=rollout_worker, args=(i, queue)) for i in range(NUM_WORKERS)]
    for p in processes:
        p.start()

    # Dummy env for shape detection and metadata
    env = make_env()
    reward_stats = RunningStats()
    obs, _ = env.reset(seed=42)
    while len(obs) == 0:
        obs, _, done, _, _ = env.step([])
        if done:
            obs, _ = env.reset(seed=42)

    obs_dim = len(obs[0])
    max_tasks = 300
    act_dim = env.num_dcs

    actor = ActorNet(obs_dim, act_dim, hidden_dim=64).to(DEVICE)
    critic = CriticNet(obs_dim, act_dim, hidden_dim=64).to(DEVICE)
    target_critic = CriticNet(obs_dim, act_dim, hidden_dim=64).to(DEVICE)
    target_critic.load_state_dict(critic.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=LR)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=LR)

    buffer = ReplayBuffer(capacity=REPLAY_SIZE, max_tasks=max_tasks, obs_dim=obs_dim)

    q_loss = policy_loss = None
    pbar = trange(TOTAL_STEPS)
    steps_collected = 0

    for global_step in pbar:
        # print(f"Global step: {global_step}")
        # === Get transitions from worker queue
        transitions_collected = 0

        while transitions_collected < TRANSITIONS_PER_ITER:
            transitions = queue.get()  # BLOCKS until data is available
            for (obs, actions, reward, next_obs, done) in transitions:
                reward_stats.update(reward)
                normalized_reward = reward_stats.normalize(reward)
                buffer.add(obs, actions, normalized_reward, next_obs, done)
                transitions_collected += 1
                steps_collected += 1
        
        # print(f"Steps collected: {steps_collected}")
        # === Training step if enough samples
        if steps_collected >= WARMUP_STEPS and len(buffer) >= BATCH_SIZE and global_step % UPDATE_FREQ == 0:
            (obs_b, act_b, rew_b, next_obs_b, done_b,
             mask_obs_b, mask_next_b) = buffer.sample(BATCH_SIZE)

            obs_b = obs_b.to(DEVICE)
            act_b = act_b.to(DEVICE)
            rew_b = rew_b.to(DEVICE)
            next_obs_b = next_obs_b.to(DEVICE)
            done_b = done_b.to(DEVICE)
            mask_obs_b = mask_obs_b.to(DEVICE)
            mask_next_b = mask_next_b.to(DEVICE)

            B, T, D = obs_b.shape
            obs_flat = obs_b.view(B*T, D)
            act_flat = act_b.view(B*T)
            mask_obs_flat = mask_obs_b.view(B*T)

            next_flat = next_obs_b.view(B*T, D)
            mask_next_flat = mask_next_b.view(B*T)

            with torch.no_grad():
                next_logits = actor(next_flat)
                next_probs = F.softmax(next_logits, dim=-1)
                next_log_probs = F.log_softmax(next_logits, dim=-1)
                q1_next, q2_next = target_critic.forward_all(next_flat)
                q_next = torch.min(q1_next, q2_next)
                v_next = (next_probs * (q_next - ALPHA * next_log_probs)).sum(dim=-1)
                v_next = v_next * mask_next_flat
                v_next = v_next.view(B, T)
                q_target = rew_b.unsqueeze(1) + GAMMA * (1 - done_b.unsqueeze(1)) * v_next

            valid_idx = act_flat >= 0
            obs_valid = obs_flat[valid_idx]
            act_valid = act_flat[valid_idx]

            q1_all, q2_all = critic(obs_valid, act_valid)

            q1_all_full = torch.zeros_like(act_flat, dtype=torch.float, device=DEVICE)
            q2_all_full = torch.zeros_like(act_flat, dtype=torch.float, device=DEVICE)
            q1_all_full[valid_idx] = q1_all
            q2_all_full[valid_idx] = q2_all

            q1_all = q1_all_full * mask_obs_flat
            q2_all = q2_all_full * mask_obs_flat

            q1_chosen = q1_all.view(B, T)
            q2_chosen = q2_all.view(B, T)
            q_target = q_target * mask_obs_b

            q1_loss = F.mse_loss(q1_chosen.view(-1)[mask_obs_b.view(-1).bool()],
                                 q_target.view(-1)[mask_obs_b.view(-1).bool()])
            q2_loss = F.mse_loss(q2_chosen.view(-1)[mask_obs_b.view(-1).bool()],
                                 q_target.view(-1)[mask_obs_b.view(-1).bool()])
            q_loss = 0.5 * (q1_loss + q2_loss)

            critic_opt.zero_grad()
            q_loss.backward()
            critic_opt.step()

            if global_step % POLICY_UPDATE_FREQ == 0:
                logits = actor(obs_flat)
                probs = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)
                q1_eval, q2_eval = critic.forward_all(obs_flat)
                q_eval = torch.min(q1_eval, q2_eval)

                policy_loss = (probs * (ALPHA * log_probs - q_eval)).sum(dim=-1)
                policy_loss = policy_loss * mask_obs_flat
                policy_loss = policy_loss.sum() / mask_obs_flat.sum()

                actor_opt.zero_grad()
                policy_loss.backward()
                actor_opt.step()

                for p, tp in zip(critic.parameters(), target_critic.parameters()):
                    tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

        if global_step % 100 == 0 and global_step > 0:
            if q_loss is not None and policy_loss is not None:
                writer.add_scalar("Loss/Q_Loss", q_loss.item(), global_step)
                writer.add_scalar("Loss/Policy_Loss", policy_loss.item(), global_step)
                pbar.write(f"[Step {global_step}] Q Loss: {q_loss.item():.3f} | Policy Loss: {policy_loss.item():.3f}")

        if global_step % SAVE_INTERVAL == 0 and global_step > 0:
            save_checkpoint(global_step, actor, critic, actor_opt, critic_opt)
            pbar.write(f"Saved checkpoint at step {global_step}")

    writer.close()
    for p in processes:
        p.terminate()
        p.join()


if __name__ == "__main__":
    train()
