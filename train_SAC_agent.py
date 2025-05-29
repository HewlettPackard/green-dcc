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
# Import all network types
from rl_components.agent_net import ActorNet, CriticNet, AttentionActorNet, AttentionCriticNet
# Import Replay Buffers
from rl_components.replay_buffer import FastReplayBuffer, SimpleReplayBuffer # Assuming SimpleReplayBuffer is created
from rewards.predefined.composite_reward import CompositeReward

from utils.checkpoint_manager import save_checkpoint # load_checkpoint is not used in this script's flow
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
    parser = argparse.ArgumentParser(description="SustainCluster Training")
    parser.add_argument("--sim-config", type=str, default="configs/env/sim_config.yaml")
    parser.add_argument("--reward-config", type=str, default="configs/env/reward_config.yaml")
    parser.add_argument("--dc-config", type=str, default="configs/env/datacenters.yaml")
    parser.add_argument("--algo-config", type=str, default="configs/env/algorithm_config.yaml")
    parser.add_argument("--tag", type=str, default="", help="Optional run tag")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-logger", type=str2bool, default=True, help="Enable logger")
    return parser.parse_args()

def make_env(sim_cfg_path, dc_cfg_path, reward_cfg_path, writer=None, logger=None):
    import pandas as pd
    from simulation.cluster_manager import DatacenterClusterManager

    sim_cfg_full = load_yaml(sim_cfg_path) # Load the full config
    sim_cfg = sim_cfg_full["simulation"]   # Extract the simulation part
    dc_cfg = load_yaml(dc_cfg_path)["datacenters"]
    reward_cfg = load_yaml(reward_cfg_path)["reward"]
    

    start = pd.Timestamp(datetime.datetime(sim_cfg["year"], sim_cfg["month"], sim_cfg["init_day"],
                                           sim_cfg["init_hour"], 0, tzinfo=datetime.timezone.utc))
    end = start + datetime.timedelta(days=sim_cfg["duration_days"])

    cluster = DatacenterClusterManager(
        config_list=dc_cfg, simulation_year=sim_cfg["year"],
        init_day=int(sim_cfg["month"] * 30.5 + sim_cfg["init_day"]), # Adjusted init_day
        init_hour=sim_cfg["init_hour"], strategy=sim_cfg["strategy"],
        tasks_file_path=sim_cfg["workload_path"],
        shuffle_datacenter_order=sim_cfg.get("shuffle_datacenters", True), # Use .get for safety
        cloud_provider=sim_cfg["cloud_provider"], logger=logger
    )
    reward_fn = CompositeReward(
        components=reward_cfg["components"],
        normalize=reward_cfg.get("normalize", False),
        freeze_stats_after_steps=reward_cfg.get("freeze_stats_after_steps", None)
    )
    # Pass the sim_cfg dict to TaskSchedulingEnv
    return TaskSchedulingEnv(
        cluster_manager=cluster, start_time=start, end_time=end,
        reward_fn=reward_fn,
        writer=writer if sim_cfg.get("use_tensorboard", False) else None,
        sim_config=sim_cfg # Pass simulation config for single_action_mode
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
    
    if logger: logger.info(f"Using device: {DEVICE}")

    eval_seed = algo_cfg.get("eval_seed", 4242) # Get from config or use a fixed one
    if logger: logger.info(f"Creating evaluation environment with seed: {eval_seed}")

    env = make_env(args.sim_config, args.dc_config, args.reward_config, writer, logger)
    eval_env = make_env(args.sim_config, args.dc_config, args.reward_config,writer=None, logger=logger)
    
    single_action_mode = env.single_action_mode
    disable_defer_action = env.disable_defer_action # Get from env
    if logger: logger.info(f"Single Action Mode: {single_action_mode}, Disable Defer Action: {disable_defer_action}")

    eval_env.single_action_mode = single_action_mode
    eval_env.disable_defer_action = disable_defer_action

    obs, _ = env.reset(seed=args.seed)
    # Initial handling for empty obs
    if single_action_mode: # obs is a single vector
        pass # No loop needed, reset should give a valid initial obs
    else: # obs is a list
        while len(obs) == 0:
            if logger: logger.info("Initial obs is empty list, stepping with no actions...")
            obs, _, done, _, _ = env.step([])
            if done: logger.error("Env terminated during init!"); obs, _ = env.reset(seed=args.seed); break

    # --- Determine Observation and Action Dimensions ---
    if single_action_mode:
        obs_dim_net = env.observation_space.shape[0] # Aggregated obs dim
        act_dim_net = env.agent_output_act_dim     # Dimension for network output (N or N+1)
    else:
        obs_dim_net = env.observation_space.shape[0] # Per-task obs dim
        act_dim_net = env.agent_output_act_dim     # Action dim per task for network (N or N+1)

    # --- Network Initialization ---
    use_layer_norm_flag = algo_cfg.get("use_layer_norm", False) # Get the flag

    use_attention = algo_cfg.get("use_attention", False) and not single_action_mode
    if logger: logger.info(f"Using attention mechanism: {use_attention}")

    if use_attention: # Only if multi-task and attention is enabled
        attn_cfg = algo_cfg.get("attention", {})
        actor = AttentionActorNet(obs_dim_net, act_dim_net, **attn_cfg).to(DEVICE)
        critic = AttentionCriticNet(obs_dim_net, act_dim_net, **attn_cfg).to(DEVICE) # Critic output is per action
        target_critic = AttentionCriticNet(obs_dim_net, act_dim_net, **attn_cfg).to(DEVICE)
    else: # MLP networks (used for single_action_mode OR if attention is false)
        actor = ActorNet(obs_dim_net, act_dim_net, hidden_dim=algo_cfg["hidden_dim"],
                         use_layer_norm=use_layer_norm_flag).to(DEVICE)
        
        critic = CriticNet(obs_dim_net, act_dim_net, hidden_dim=algo_cfg["hidden_dim"],
                           use_layer_norm=use_layer_norm_flag).to(DEVICE) # Critic output is per action
        
        target_critic = CriticNet(obs_dim_net, act_dim_net, hidden_dim=algo_cfg["hidden_dim"],
                                  use_layer_norm=use_layer_norm_flag).to(DEVICE)
    target_critic.load_state_dict(critic.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=float(algo_cfg["actor_learning_rate"]))
    critic_opt = torch.optim.Adam(critic.parameters(), lr=float(algo_cfg["critic_learning_rate"]))

    # --- Replay Buffer ---
    if single_action_mode:
        buffer = SimpleReplayBuffer( # Assumes you create this class
            capacity=algo_cfg["replay_buffer_size"],
            obs_dim=obs_dim_net,
            # act_dim is implicitly 1 for discrete actions
        )
        if logger: logger.info(f"Using SimpleReplayBuffer with obs_dim: {obs_dim_net}")
    else:
        buffer = FastReplayBuffer(
            capacity=algo_cfg["replay_buffer_size"],
            max_tasks=algo_cfg["max_tasks"],
            obs_dim=obs_dim_net # This is per-task obs_dim
        )
        if logger: logger.info(f"Using FastReplayBuffer with max_tasks: {algo_cfg['max_tasks']}, obs_dim_per_task: {obs_dim_net}")


    # --- Training Loop Variables ---
    reward_stats = RunningStats()
    episode_reward = 0
    episode_steps = 0
    best_eval_reward = float("-inf") # Track best *evaluation* reward

    episode_reward_buffer = deque(maxlen=10)
    best_avg_reward = float("-inf")
    
    q_loss, policy_loss = None, None # Renamed to avoid conflict with torch.nn.functional

    pbar = trange(algo_cfg["total_steps"])
    for global_step in pbar:
        # --- Action Sampling ---
        if single_action_mode:
            # obs is already a single vector (potentially zero vector if no tasks)
            # If no tasks, agent still sees a zero vector and takes an action (e.g. defer)
            # env.step() handles the "no tasks to apply action to" case internally.
            obs_tensor_actor = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE) # Add batch dim
            if global_step < algo_cfg["warmup_steps"]:
                action_scalar = env.action_space.sample() # Samples a single int
            else:
                with torch.no_grad():
                    logits = actor(obs_tensor_actor) # Expects [B, D_obs_agg] -> [B, D_act]
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    action_scalar = dist.sample().item() # Get single int action
            actions_to_env = action_scalar # Pass single action to env
            num_actions_taken = 1 if len(env.current_tasks) > 0 else 0 # Log if action applied
        else: # Multi-task mode
            if len(obs) == 0: # No tasks, obs is empty list
                actions_list = []
            elif global_step < algo_cfg["warmup_steps"]:
                actions_list = [np.random.randint(act_dim_net) for _ in obs]
            else:
                if obs: # Ensure obs is not empty before trying to stack
                    obs_numpy_array = np.array(obs, dtype=np.float32)
                    obs_tensor_actor = torch.from_numpy(obs_numpy_array).to(DEVICE) # [k_t, D_obs_per_task]
                else: # Should not happen if len(obs) == 0 check is above, but defensive
                    obs_tensor_actor = torch.empty((0, obs_dim_net), device=DEVICE, dtype=torch.float32) # [k_t, D_obs_per_task]
                with torch.no_grad():
                    logits = actor(obs_tensor_actor) # Expects [k_t, D_obs_per_task] -> [k_t, D_act]
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    actions_list = dist.sample().cpu().numpy().tolist()
                    
                    assert all(0 <= a < act_dim_net for a in actions_list), f"Sampled invalid action: {actions_list}"

            actions_to_env = actions_list # Pass list of actions to env
            num_actions_taken = len(actions_list)

        # --- Step Environment ---
        next_obs_env, reward, done, truncated, info = env.step(actions_to_env)
        reward_stats.update(reward)
        normalized_reward = reward_stats.normalize(reward)

        # --- Store in Buffer ---
        if num_actions_taken > 0 or single_action_mode: # Store even if no tasks in single_action_mode
            if single_action_mode:
                buffer.add(obs, actions_to_env, normalized_reward, next_obs_env, done or truncated)
            else: # Multi-task mode
                buffer.add(obs, actions_to_env, normalized_reward, next_obs_env, done or truncated)

        obs = next_obs_env
        episode_reward += reward
        episode_steps += 1

        # --- Episode End Logic (similar, but obs reset differs) ---
        if done or truncated:
            avg_ep_reward = episode_reward / episode_steps if episode_steps > 0 else 0.0
            episode_reward_buffer.append(avg_ep_reward)
            if logger: logger.info(f"[Episode End] Step: {global_step}, Reward: {episode_reward:.2f}, Avg Reward: {avg_ep_reward:.2f}")
            writer.add_scalar("Reward/Episode", avg_ep_reward, global_step)
            pbar.write(f"Ep. Reward: {avg_ep_reward:.2f} (steps: {episode_steps})")

            obs, _ = env.reset(seed=args.seed + global_step // 1000) # Vary seed less frequently
            episode_reward = 0
            episode_steps = 0

            if len(episode_reward_buffer) == 10:
                avg10_reward = np.mean(episode_reward_buffer)
                writer.add_scalar("Reward/Avg10", avg10_reward, global_step)
                pbar.write(f"Avg10 Reward: {avg10_reward:.2f}")
                if avg10_reward > best_avg_reward:
                    best_avg_reward = avg10_reward
                    current_extra_info = {
                        'single_action_mode': single_action_mode,
                        'disable_defer_action': disable_defer_action,
                        'use_attention': algo_cfg.get("use_attention", False), # Get from loaded algo_cfg
                        'obs_dim': obs_dim_net, # Use the obs_dim of the network
                        'act_dim': act_dim_net,  # Use the act_dim of the network
                        'hidden_dim': algo_cfg.get("hidden_dim", 64), # For MLPs
                        'use_layer_norm': algo_cfg.get("use_layer_norm", False), # For MLPs
                    }
                    save_checkpoint(
                        global_step, actor, critic, actor_opt, critic_opt,
                        ckpt_dir,
                        is_best=True,
                        extra_info=current_extra_info # Pass extra_info here
                    )
                    pbar.write(f"New best avg reward: {best_avg_reward:.2f}")
                    if logger: logger.info(f"[BEST] New best avg reward: {best_avg_reward:.2f}")


        # --- RL Updates ---
        if global_step >= algo_cfg["warmup_steps"] and global_step % algo_cfg["update_frequency"] == 0:
            if len(buffer) < algo_cfg["batch_size"]:
                continue

            if single_action_mode:
                # --- SAC Update for Single Action Mode ---
                obs_b, act_b, rew_b, next_obs_b, done_b = buffer.sample(algo_cfg["batch_size"])
                obs_b = obs_b.to(DEVICE)         # [B, obs_dim_aggregated]
                act_b = act_b.to(DEVICE)         # [B] (scalar actions)
                rew_b = rew_b.to(DEVICE).unsqueeze(1)        # [B, 1]
                next_obs_b = next_obs_b.to(DEVICE)# [B, obs_dim_aggregated]
                done_b = done_b.to(DEVICE).unsqueeze(1)      # [B, 1]

                with torch.no_grad():
                    next_logits = actor(next_obs_b) # [B, act_dim]
                    next_probs = F.softmax(next_logits, dim=-1)
                    next_log_probs = F.log_softmax(next_logits, dim=-1) # [B, act_dim]
                    q1_next, q2_next = target_critic.forward_all(next_obs_b) # [B, act_dim], [B, act_dim]
                    q_next = torch.min(q1_next, q2_next) # [B, act_dim]
                    v_next = (next_probs * (q_next - algo_cfg["alpha"] * next_log_probs)).sum(dim=-1, keepdim=True) # [B, 1]
                    q_target = rew_b + algo_cfg["gamma"] * (1.0 - done_b) * v_next # [B, 1]

                # Critic loss
                # q1_all/q2_all are [B, act_dim]. gather for chosen actions
                q1_pred_all, q2_pred_all = critic.forward_all(obs_b) # [B, act_dim]
                q1_pred = q1_pred_all.gather(1, act_b.long().unsqueeze(-1)) # [B, 1]
                q2_pred = q2_pred_all.gather(1, act_b.long().unsqueeze(-1)) # [B, 1]

                q1_loss = F.mse_loss(q1_pred, q_target)
                q2_loss = F.mse_loss(q2_pred, q_target)
                q_loss = 0.5 * (q1_loss + q2_loss)

                critic_opt.zero_grad()
                q_loss.backward()
                critic_opt.step()

                # Actor loss
                if global_step % algo_cfg["policy_update_frequency"] == 0:
                    logits = actor(obs_b) # [B, act_dim]
                    probs = F.softmax(logits, dim=-1)
                    log_probs = F.log_softmax(logits, dim=-1) # [B, act_dim]
                    q1_eval, q2_eval = critic.forward_all(obs_b) # [B, act_dim]
                    q_eval = torch.min(q1_eval, q2_eval) # [B, act_dim]

                    policy_loss = (probs * (algo_cfg["alpha"] * log_probs - q_eval.detach())).sum(dim=-1).mean() # Scalar
                    entropy = torch.distributions.Categorical(logits=logits).entropy().mean()

                    actor_opt.zero_grad()
                    policy_loss.backward()
                    actor_opt.step()

                    # Target critic update
                    for p, tp in zip(critic.parameters(), target_critic.parameters()):
                        tp.data.copy_(algo_cfg["tau"] * p.data + (1 - algo_cfg["tau"]) * tp.data)
            else:
                # --- Original SAC Update for Multi-Task Mode (with masking) ---
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
                if (masked_actions > act_dim_net).any() or (masked_actions < 0).any():
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

                # Critic loss
                # compute Q values for actual (obs, action) pairs
                valid_idx = (act_flat >= 0) & (act_flat < act_dim_net)
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

        # --- Logging (common part) ---
        if global_step % algo_cfg["log_interval"] == 0 and q_loss and policy_loss:
            if q_loss is not None and policy_loss is not None:
                writer.add_scalar("Loss/Q_Loss", q_loss.item(), global_step)
                writer.add_scalar("Loss/Policy_Loss", policy_loss.item(), global_step)
                if logger: logger.info(f"[Step {global_step}] Q Loss: {q_loss.item():.4f} | Policy Loss: {policy_loss.item():.4f}")
                pbar.set_description(f"Q Loss: {q_loss.item():.3f} P Loss: {policy_loss.item():.3f}")
            else:
                if logger: logger.info(f"[Step {global_step}] Skipping logging — losses not available yet.")

        if global_step > 0 and global_step % algo_cfg["save_interval"] == 0:
            current_extra_info = {
                        'single_action_mode': single_action_mode,
                        'disable_defer_action': disable_defer_action,
                        'use_attention': algo_cfg.get("use_attention", False), # Get from loaded algo_cfg
                        'obs_dim': obs_dim_net, # Use the obs_dim of the network
                        'act_dim': act_dim_net,  # Use the act_dim of the network
                        'hidden_dim': algo_cfg.get("hidden_dim", 64), # For MLPs
                        'use_layer_norm': algo_cfg.get("use_layer_norm", False), # For MLPs
                    }
            save_checkpoint(
                global_step, actor, critic, actor_opt, critic_opt,
                ckpt_dir,
                is_best=False, # This is a periodic save, not necessarily the best
                extra_info=current_extra_info
            )
            pbar.write(f"Saved periodic checkpoint at step {global_step} with extra_info.")
            if logger: logger.info(f"Saved periodic checkpoint at step {global_step} with extra_info.")


        # --- Periodic Evaluation and Saving Best Evaluation Model ---
        if global_step > 0 and global_step % algo_cfg["eval_frequency"] == 0:
            if logger: logger.info(f"--- Starting Evaluation at step {global_step} ---")
            actor.eval() # Set actor to evaluation mode
            total_eval_reward = 0
            total_eval_steps = 0
            num_eval_episodes_done = 0

            eval_obs, _ = eval_env.reset(seed=eval_seed) # Use fixed seed for eval_env
            if single_action_mode:
                if eval_obs is None: # Should be initialized correctly
                    if logger: logger.error("Eval env reset to None in single_action_mode")
                    # Handle this case, maybe reset again or skip eval
            else: # Multi-task
                while len(eval_obs) == 0 and num_eval_episodes_done < algo_cfg["eval_episodes"]:
                    eval_obs, _, eval_done, _, _ = eval_env.step([])
                    if eval_done:
                        eval_obs, _ = eval_env.reset(seed=eval_seed + num_eval_episodes_done) # Next eval ep
                        num_eval_episodes_done += 1


            for eval_ep in range(algo_cfg["eval_episodes"]):
                ep_eval_reward = 0
                ep_eval_steps = 0
                eval_done_ep = False
                eval_truncated_ep = False

                # Use the same reset logic for potentially empty initial obs
                if single_action_mode:
                    if eval_obs is None and eval_ep > 0 : # if it was None from previous done
                        eval_obs, _ = eval_env.reset(seed=eval_seed + eval_ep)
                else:
                    is_initial_step_of_ep = True
                    while len(eval_obs) == 0:
                        if not is_initial_step_of_ep: # Avoid infinite loop if reset always gives empty
                            logger.warning("Eval episode started with empty obs repeatedly.")
                            break
                        eval_obs, _, eval_done_ep, eval_truncated_ep, _ = eval_env.step([])
                        is_initial_step_of_ep = False
                        if eval_done_ep or eval_truncated_ep: break
                    if eval_done_ep or eval_truncated_ep: continue # Skip to next eval episode

                while not (eval_done_ep or eval_truncated_ep):
                    with torch.no_grad():
                        if single_action_mode:
                            eval_obs_tensor = torch.FloatTensor(eval_obs).unsqueeze(0).to(DEVICE)
                            eval_logits = actor(eval_obs_tensor)
                            eval_probs = F.softmax(eval_logits, dim=-1)
                            eval_action_scalar = torch.distributions.Categorical(eval_probs).sample().item()
                            eval_actions_to_env = eval_action_scalar
                        else: # Multi-task
                            if len(eval_obs) == 0:
                                eval_actions_list = []
                            else:
                                # eval_obs_tensor = torch.FloatTensor(eval_obs).to(DEVICE)

                                if eval_obs: # Ensure obs is not empty before trying to stack
                                    eval_obs_numpy_array = np.array(eval_obs, dtype=np.float32)
                                    eval_obs_tensor_actor = torch.from_numpy(eval_obs_numpy_array).to(DEVICE) # [k_t, D_obs_per_task]
                                else: # Should not happen if len(obs) == 0 check is above, but defensive
                                    eval_obs_tensor_actor = torch.empty((0, obs_dim_net), device=DEVICE, dtype=torch.float32)
                    
                                eval_logits = actor(eval_obs_tensor_actor)
                                eval_probs = F.softmax(eval_logits, dim=-1)
                                eval_actions_list = torch.distributions.Categorical(eval_probs).sample().cpu().numpy().tolist()
                            eval_actions_to_env = eval_actions_list

                    eval_next_obs, eval_reward, eval_done_ep, eval_truncated_ep, _ = eval_env.step(eval_actions_to_env)
                    ep_eval_reward += eval_reward
                    ep_eval_steps += 1
                    eval_obs = eval_next_obs

                    if eval_done_ep or eval_truncated_ep:
                        break # Inner loop for episode steps

                total_eval_reward += ep_eval_reward
                total_eval_steps += ep_eval_steps
                if logger: logger.info(f"Eval Ep {eval_ep+1}/{algo_cfg['eval_episodes']} Reward: {ep_eval_reward:.2f} ({ep_eval_steps} steps)")
                # Reset for next eval episode, use a consistent seed pattern for eval episodes
                if eval_ep < algo_cfg["eval_episodes"] - 1:
                    eval_obs, _ = eval_env.reset(seed=eval_seed + eval_ep + 1)
                    # Handle empty obs after reset for eval
                    if single_action_mode:
                        if eval_obs is None: logger.error("Eval env reset to None during eval loop")
                    else:
                        is_initial_step_of_ep = True
                        while len(eval_obs) == 0:
                            if not is_initial_step_of_ep: break
                            eval_obs, _, eval_done_ep, eval_truncated_ep, _ = eval_env.step([])
                            is_initial_step_of_ep = False
                            if eval_done_ep or eval_truncated_ep: break
                        if eval_done_ep or eval_truncated_ep: break # Break outer eval ep loop

            avg_eval_reward = total_eval_reward / algo_cfg["eval_episodes"] if algo_cfg["eval_episodes"] > 0 else 0.0
            writer.add_scalar("Reward/Eval_AverageEpisode", avg_eval_reward, global_step)
            if logger: logger.info(f"--- Evaluation at step {global_step} COMPLETE --- Avg Reward: {avg_eval_reward:.2f}")
            pbar.write(f"Evaluation Avg Reward: {avg_eval_reward:.2f}")

            if avg_eval_reward > best_eval_reward:
                best_eval_reward = avg_eval_reward
                if logger: logger.info(f"[BEST EVAL] New best evaluation reward: {best_eval_reward:.2f}. Saving model...")
                current_extra_info = {
                        'single_action_mode': single_action_mode,
                        'disable_defer_action': disable_defer_action,
                        'use_attention': algo_cfg.get("use_attention", False), # Get from loaded algo_cfg
                        'obs_dim': obs_dim_net, # Use the obs_dim of the network
                        'act_dim': act_dim_net,  # Use the act_dim of the network
                        'hidden_dim': algo_cfg.get("hidden_dim", 64), # For MLPs
                        'use_layer_norm': algo_cfg.get("use_layer_norm", False), # For MLPs
                    }
                if use_attention:
                    attn_cfg = algo_cfg.get("attention", {})
                    current_extra_info.update({
                        'attn_embed_dim': attn_cfg.get("embed_dim", 128),
                        'attn_num_heads': attn_cfg.get("num_heads", 4),
                        'attn_num_layers': attn_cfg.get("num_attention_layers", 2),
                        'attn_dropout': attn_cfg.get("dropout", 0.1),
                        'attn_use_layer_norm': attn_cfg.get("use_layer_norm_flag", True)
                    })
                save_checkpoint(global_step, actor, critic, actor_opt, critic_opt,
                                ckpt_dir, filename="best_eval_checkpoint.pth", # Specific name
                                extra_info=current_extra_info)
            actor.train() # Set actor back to training mode
        
    writer.close()
    if logger: logger.info("Training finished.")

if __name__ == "__main__":
    train()