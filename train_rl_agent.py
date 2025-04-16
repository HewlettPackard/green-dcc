import torch
import torch.nn.functional as F
import numpy as np
from tqdm import trange
import logging
import os
from collections import deque
import argparse
import datetime

from envs.task_scheduling_env import TaskSchedulingEnv
from rl_components.agent_net import ActorNet, CriticNet
from rl_components.replay_buffer import FastReplayBuffer
from rewards.predefined.composite_reward import CompositeReward

from utils.checkpoint_manager import save_checkpoint, load_checkpoint
from utils.config_loader import load_yaml
from utils.config_logger import setup_logger
from torch.utils.tensorboard import SummaryWriter


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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
    
def parse_args():
    parser = argparse.ArgumentParser(description="GreenDCC Training")
    parser.add_argument("--sim-config", type=str, default="configs/env/sim_config.yaml")
    parser.add_argument("--reward-config", type=str, default="configs/env/reward_config.yaml")
    parser.add_argument("--dc-config", type=str, default="configs/env/datacenters.yaml")
    parser.add_argument("--algo-config", type=str, default="configs/env/algorithm_config.yaml")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--tag", type=str, default="", help="Optional run tag")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-logger", type=str2bool, default=True, help="Enable logger")
    return parser.parse_args()


def make_env(sim_cfg_path, dc_cfg_path, reward_cfg_path, writer=None, logger=None):
    import pandas as pd
    import datetime
    from simulation.cluster_manager import DatacenterClusterManager

    sim_cfg = load_yaml(sim_cfg_path)["simulation"]
    dc_cfg = load_yaml(dc_cfg_path)["datacenters"]
    reward_cfg = load_yaml(reward_cfg_path)["reward"]

    start = pd.Timestamp(datetime.datetime(sim_cfg["year"], sim_cfg["month"], sim_cfg["init_day"],
                                           sim_cfg["init_hour"], 0, tzinfo=datetime.timezone.utc))
    end = start + datetime.timedelta(days=sim_cfg["duration_days"])

    cluster = DatacenterClusterManager(
        config_list=dc_cfg,
        simulation_year=sim_cfg["year"],
        init_day=int(sim_cfg["month"] * 30.5),
        init_hour=sim_cfg["init_hour"],
        strategy=sim_cfg["strategy"],
        tasks_file_path=sim_cfg["workload_path"],
        shuffle_datacenter_order=sim_cfg["shuffle_datacenters"],
        cloud_provider=sim_cfg["cloud_provider"],
        logger=logger
    )

    reward_fn = CompositeReward(
        components=reward_cfg["components"],
        normalize=reward_cfg.get("normalize", False)
    )

    return TaskSchedulingEnv(
        cluster_manager=cluster,
        start_time=start,
        end_time=end,
        reward_fn=reward_fn,
        writer=writer if sim_cfg.get("use_tensorboard", False) else None
    )


def train():
    args = parse_args()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.tag}_{timestamp}" if args.tag else f"{timestamp}"

    # Save logs separately from tensorboard
    log_dir = f"logs/train_{run_id}"          # <--- debug logs
    tb_dir = f"runs/train_{run_id}"           # <--- tensorboard
    ckpt_dir = f"checkpoints/train_{run_id}"  # <--- checkpoints

    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)   # <-- use only for TensorBoard

    print(f"Enable logger: {args.enable_logger}")
    logger = setup_logger(log_dir, enable_logger=args.enable_logger)

    algo_cfg = load_yaml(args.algo_config)["algorithm"]
    if algo_cfg["device"] == "auto":
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        DEVICE = torch.device(algo_cfg["device"])

    env = make_env(args.sim_config, args.dc_config, args.reward_config, writer, logger)

    obs, _ = env.reset(seed=args.seed)
    while len(obs) == 0:
        obs, _, done, _, _ = env.step([])
        if done:
            obs, _ = env.reset(seed=args.seed)

    obs_dim = len(obs[0])
    act_dim = env.num_dcs + 1

    actor = ActorNet(obs_dim, act_dim, hidden_dim=algo_cfg["hidden_dim"]).to(DEVICE)
    critic = CriticNet(obs_dim, act_dim, hidden_dim=algo_cfg["hidden_dim"]).to(DEVICE)
    target_critic = CriticNet(obs_dim, act_dim, hidden_dim=algo_cfg["hidden_dim"]).to(DEVICE)
    target_critic.load_state_dict(critic.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=float(algo_cfg["actor_learning_rate"]))
    critic_opt = torch.optim.Adam(critic.parameters(), lr=float(algo_cfg["critic_learning_rate"]))

    buffer = FastReplayBuffer(
        capacity=algo_cfg["replay_buffer_size"],
        max_tasks=algo_cfg["max_tasks"],
        obs_dim=obs_dim
    )

    reward_stats = RunningStats()
    episode_reward = 0
    episode_steps = 0
    episode_reward_buffer = deque(maxlen=10)
    best_avg_reward = float("-inf")
    q_loss = None
    policy_loss = None

    pbar = trange(algo_cfg["total_steps"])

    for global_step in pbar:
        obs_tensor = torch.FloatTensor(obs).to(DEVICE)
        if len(obs) == 0:
            actions = []
        elif global_step < algo_cfg["warmup_steps"]:
            actions = [np.random.randint(act_dim) for _ in obs]
        else:
            with torch.no_grad():
                logits = actor(obs_tensor)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                actions = dist.sample().cpu().numpy().tolist()
                writer.add_histogram("Actor/logits", logits, global_step)
                writer.add_histogram("Actor/probs", probs, global_step)
                
                assert all(0 <= a < act_dim for a in actions), f"Sampled invalid action: {actions}"
        
        # 6) Step environment
        next_obs, reward, done, truncated, _ = env.step(actions)
        reward_stats.update(reward)
        normalized_reward = reward_stats.normalize(reward)

        if len(actions) > 0:
            buffer.add(obs, actions, normalized_reward, next_obs, done or truncated)

        obs = next_obs
        episode_reward += reward
        episode_steps += 1

        if done or truncated:
            avg = episode_reward / episode_steps
            episode_reward_buffer.append(avg)
            
            if logger:
                logger.info(f"[Episode End] total_reward={avg:.2f}")
                
            writer.add_scalar("Reward/Episode", avg, global_step)
            pbar.write(f"Episode reward: {avg:.2f} (steps: {episode_steps})")
            obs, _ = env.reset(seed=args.seed+global_step)
            episode_reward = 0
            episode_steps = 0
            if len(episode_reward_buffer) == 10:
                avg_reward = np.mean(episode_reward_buffer)
                writer.add_scalar("Reward/Avg10", avg_reward, global_step)

                pbar.write(f"Avg reward: {avg_reward:.2f}")
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    save_checkpoint(global_step, actor, critic, actor_opt, critic_opt, ckpt_dir, best=True)
                    pbar.write(f"[BEST] Saved checkpoint at step {global_step} (avg10 reward={avg_reward:.2f})")

        # 7) RL updates
        if global_step >= algo_cfg["warmup_steps"] and global_step % algo_cfg["update_frequency"] == 0:
            if len(buffer) < algo_cfg["batch_size"]:
                continue
            (obs_b, act_b, rew_b, next_obs_b, done_b,
             mask_obs_b, mask_next_b) = buffer.sample(algo_cfg["batch_size"])
        
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
                v_next = (next_probs * (q_next - algo_cfg["alpha"] * next_log_probs)).sum(dim=-1)
                v_next = v_next * mask_next_flat
                # reshape to (B, T)
                v_next = v_next.view(B, T)

                # q_target shape: (B, T)
                q_target = rew_b.unsqueeze(1) + algo_cfg["gamma"] * (1 - done_b.unsqueeze(1)) * v_next

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
            if global_step % algo_cfg["policy_update_frequency"] == 0:
                logits = actor(obs_flat)
                probs = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)
                q1_eval, q2_eval = critic.forward_all(obs_flat)
                q_eval = torch.min(q1_eval, q2_eval)

                # shape of probs, q_eval, log_probs => (B*T, act_dim)
                # compute the policy loss with mask
                policy_loss = (probs * (algo_cfg["alpha"] * log_probs - q_eval)).sum(dim=-1)
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
                    tp.data.copy_(algo_cfg["tau"] * p.data + (1 - algo_cfg["tau"]) * tp.data)

        if global_step % algo_cfg["log_interval"] == 0 and q_loss and policy_loss:
            if q_loss is not None and policy_loss is not None:
                writer.add_scalar("Loss/Q_Loss", q_loss.item(), global_step)
                writer.add_scalar("Loss/Policy_Loss", policy_loss.item(), global_step)

                if logger:
                    logger.info(f"[Step {global_step}] Q Loss: {q_loss.item():.3f} | Policy Loss: {policy_loss.item():.3f}")
                pbar.write(f"[Step {global_step}] Q Loss: {q_loss.item():.3f} | Policy Loss: {policy_loss.item():.3f}")
            else:
                if logger:
                    logger.info(f"[Step {global_step}] Skipping logging — losses not available yet.")
        

        if global_step % algo_cfg["save_interval"] == 0 and global_step > 0:
            save_checkpoint(global_step, actor, critic, actor_opt, critic_opt, ckpt_dir)
            pbar.write(f"Saved checkpoint at step {global_step}")

    writer.close()


if __name__ == "__main__":
    train()