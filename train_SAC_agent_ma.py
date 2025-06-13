import torch
import torch.nn.functional as F
import numpy as np
from tqdm import trange
import logging
import os
from collections import deque
import argparse
import datetime
import pandas as pd

from envs.sustaincluster_ma_env import SustainClusterMAEnv
from simulation.cluster_manager_ma import DatacenterClusterManagerMA
# Import all network types
from rl_components.agent_net_ma import ManagerActor, ManagerCritic, WorkerActor, WorkerCritic
# Import Replay Buffers
from rl_components.replay_buffer_ma import ManagerReplayBuffer, WorkerReplayBuffer
from rewards.predefined.composite_reward import CompositeReward
from utils.marl_utils import D_META_MANAGER, D_META_WORKER
from utils.checkpoint_manager_ma import save_checkpointMA # load_checkpoint is not used in this script's flow
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
    parser = argparse.ArgumentParser(description="Hierachical Multi-agent SustainCluster Training")
    parser.add_argument("--sim-config", type=str, default="configs/env/sim_config_ma.yaml")
    parser.add_argument("--reward-config", type=str, default="configs/env/reward_config.yaml")
    parser.add_argument("--dc-config", type=str, default="configs/env/datacenters_ma.yaml")
    parser.add_argument("--algo-config", type=str, default="configs/env/algorithm_config_ma.yaml")
    parser.add_argument("--tag", type=str, default="", help="Optional run tag")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-logger", type=str2bool, default=True, help="Enable logger")
    return parser.parse_args()


def make_env(sim_cfg_path, dc_cfg_path, reward_cfg_path, writer=None, logger=None):

    sim_cfg_full = load_yaml(sim_cfg_path) # Load the full config
    sim_cfg = sim_cfg_full["simulation"]   # Extract the simulation part
    dc_cfg = load_yaml(dc_cfg_path)["datacenters"]
    reward_cfg = load_yaml(reward_cfg_path)["reward"]

    for cfg in dc_cfg:
        cfg.setdefault("simulation_year", sim_cfg["year"])
    start = pd.Timestamp(datetime.datetime(sim_cfg["year"], sim_cfg["month"], sim_cfg["init_day"],
                                           sim_cfg["init_hour"], 0, tzinfo=datetime.timezone.utc))
    end = start + datetime.timedelta(days=sim_cfg["duration_days"])

    cluster = DatacenterClusterManagerMA(
        config_list=dc_cfg, 
        simulation_year=sim_cfg["year"],
        tasks_file_path=sim_cfg["workload_path"],
        cloud_provider=sim_cfg["cloud_provider"], 
        max_total_options=sim_cfg["max_total_options"],
        logger=logger,
    )

    reward_fn = CompositeReward(components=reward_cfg["components"], normalize=False)

    return SustainClusterMAEnv(cluster_manager_ma=cluster, start_time=start, end_time=end,
        reward_fn=reward_fn, logger = logger)

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
    logger = setup_logger(log_dir, enable_logger=args.enable_logger)

    algo_cfg = load_yaml(args.algo_config)["algorithm"]
    torch.manual_seed(args.seed) 
    np.random.seed(args.seed)
    train_seed = args.seed
    eval_seed = args.seed + 1000


    if algo_cfg["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(algo_cfg["device"])
    
    if logger: logger.info(f"Using device: {device}")

    env = make_env(args.sim_config, args.dc_config, args.reward_config, writer, logger)
    eval_env =  make_env(args.sim_config, args.dc_config, args.reward_config, writer, logger)

    obs_dict, _ = env.reset(seed = train_seed)

    D_GLOBAL = 4
    D_OPT = env.cluster_manager_ma.D_OPTION_FEAT
    MAX_OPT = env.cluster_manager_ma.max_total_options
    D_LOCAL_WORKER = obs_dict[f"worker_{env._dc_ids[0]}"]["obs_local_dc_i_for_worker"].shape[0]

    # --- Network Initialization ---

    mgr_actor   = ManagerActor(D_META_MANAGER, D_GLOBAL, D_OPT, MAX_OPT).to(device)
    mgr_critic  = ManagerCritic(D_META_MANAGER, D_GLOBAL, D_OPT, MAX_OPT).to(device)
    mgr_target_critic = ManagerCritic(D_META_MANAGER, D_GLOBAL, D_OPT, MAX_OPT).to(device)
    mgr_target_critic.load_state_dict(mgr_critic.state_dict())

    wrk_actor   = WorkerActor(D_META_WORKER, D_LOCAL_WORKER, D_GLOBAL).to(device)
    wrk_critic  = WorkerCritic(D_META_WORKER, D_LOCAL_WORKER, D_GLOBAL).to(device)
    wrk_target_critic = WorkerCritic(D_META_WORKER, D_LOCAL_WORKER, D_GLOBAL).to(device)
    wrk_target_critic.load_state_dict(wrk_critic.state_dict())

    mgr_actor_opt = torch.optim.Adam(mgr_actor.parameters(), lr=float(algo_cfg["actor_learning_rate"]))
    mgr_critic_opt = torch.optim.Adam(mgr_critic.parameters(),lr=float(algo_cfg["critic_learning_rate"]))
    wrk_actor_opt = torch.optim.Adam(wrk_actor.parameters(), lr=float(algo_cfg["actor_learning_rate"]))
    wrk_critic_opt = torch.optim.Adam(wrk_critic.parameters(),lr=float(algo_cfg["critic_learning_rate"]))

    # --- Replay Buffer ---

    mgr_buffer = ManagerReplayBuffer(capacity = algo_cfg["replay_buffer_size"], 
                                    D_emb_meta_manager = D_META_MANAGER, 
                                    D_global = D_GLOBAL, 
                                    D_option_feat = D_OPT, 
                                    max_total_options = MAX_OPT)
    
    wrk_buffer = WorkerReplayBuffer(capacity = algo_cfg["replay_buffer_size"], 
                                   D_emb_meta_worker = D_META_WORKER,
                                   D_emb_local_worker = D_LOCAL_WORKER,
                                   D_global = D_GLOBAL)
    
   # --- Training Loop Variables ---
    stats = RunningStats()
    episode_reward, episode_steps = 0.0, 0
    best_eval_reward = float("-inf") # Track best *evaluation* reward
    episode_reward_buffer = deque(maxlen=10)
    best_avg_reward = float("-inf")
    pbar = trange(algo_cfg["total_steps"])
    critic_loss_m, critic_loss_w, actor_loss_m, actor_loss_w= None, None, None, None

    for global_step in pbar:
        
        meta_m, opt_m, mask_m, glob_m= [], [], [], []

        for dc in env._dc_ids:
            o_mgr = obs_dict[f"manager_{dc}"]
            meta_m.append(o_mgr["obs_manager_meta_task_i"])
            opt_m.append(o_mgr["obs_all_options_set_padded"])
            mask_m.append(o_mgr["all_options_padding_mask"])
            glob_m.append(o_mgr["global_context"])

        # ------------ Manager tensors ------------
        meta_m = torch.from_numpy(np.asarray(meta_m, dtype=np.float32)).to(device)   # [N_dc, D_meta_m]
        opt_m  = torch.from_numpy(np.asarray(opt_m,  dtype=np.float32)).to(device)   # [N_dc, max_opt, D_opt]
        mask_m = torch.from_numpy(np.asarray(mask_m, dtype=np.bool_  )).to(device)   # [N_dc, max_opt]
        glob_m = torch.from_numpy(np.asarray(glob_m, dtype=np.float32)).to(device)   # [N_dc, D_global]
    
        with torch.no_grad():
            act_m, _, _ = mgr_actor.sample_action(meta_m, glob_m, opt_m, mask_m)

        mgr_act = {dc: act_m[i].item() for i, dc in enumerate(env._dc_ids)}
        obs_after_mgr = env.manager_step(mgr_act)

        meta_w, local_w, glob_w = [], [], []
        for dc in env._dc_ids:
            o_wrk = obs_after_mgr[f"worker_{dc}"]
            meta_w.append( o_wrk["obs_worker_meta_task_i"] )
            local_w.append(o_wrk["obs_local_dc_i_for_worker"])
            glob_w.append( o_wrk["global_context"] )

        # ------------ Worker tensors ------------
        meta_w  = torch.from_numpy(np.asarray(meta_w,  dtype=np.float32)).to(device) # [N_dc, D_meta_w]
        local_w = torch.from_numpy(np.asarray(local_w, dtype=np.float32)).to(device) # [N_dc, D_local_w]
        glob_w  = torch.from_numpy(np.asarray(glob_w,  dtype=np.float32)).to(device) # [N_dc, D_global]

        with torch.no_grad():
            act_w, _, _ = wrk_actor.sample_action(meta_w, local_w, glob_w)
        wrk_act = {dc: int(act_w[i].item()) for i, dc in enumerate(env._dc_ids)}
        env.worker_step(wrk_act)

        next_obs, rewards, dones, truncated, infos = env.env_step()
        done_flag = dones["__all__"] or truncated["__all__"]
        global_reward = rewards[next(iter(rewards))]
        stats.update(global_reward)
        norm_reward = stats.normalize(global_reward)

        # === store transition ===

        for i, dc in enumerate(env._dc_ids):
            # Manager
            mgr_buffer.add(
                meta_m[i].cpu().numpy(), glob_m[i].cpu().numpy(),
                opt_m[i].cpu().numpy(),   mask_m[i].cpu().numpy(),
                act_m[i].item(),          norm_reward, done_flag,
                next_obs[f"manager_{dc}"]["obs_manager_meta_task_i"],
                next_obs[f"manager_{dc}"]["global_context"],
                next_obs[f"manager_{dc}"]["obs_all_options_set_padded"],
                next_obs[f"manager_{dc}"]["all_options_padding_mask"],
            )
            # Worker
            wrk_buffer.add(
                meta_w[i].cpu().numpy(), local_w[i].cpu().numpy(), glob_w[i].cpu().numpy(),
                int(act_w[i].item()),         norm_reward, done_flag,
                next_obs[f"worker_{dc}"]["obs_worker_meta_task_i"],
                next_obs[f"worker_{dc}"]["obs_local_dc_i_for_worker"],
                next_obs[f"worker_{dc}"]["global_context"],
            )
        obs_dict = next_obs
        episode_reward += global_reward
        episode_steps += 1

        if done_flag:
            avg_ep_reward = episode_reward/episode_steps if episode_steps > 0 else 0.0
            episode_reward_buffer.append(avg_ep_reward)
            writer.add_scalar("Reward/Episode", avg_ep_reward, global_step)
            if logger: logger.info(f"[Episode End] Step: {global_step}, Reward: {episode_reward:.2f}, Avg Reward: {avg_ep_reward:.2f}")
            pbar.write(f"Ep. Reward: {avg_ep_reward:.2f} (steps: {episode_steps})")
            obs_dict = next_obs if not done_flag else env.reset(seed=args.seed + global_step // 1000)[0]
            episode_reward = 0
            episode_steps = 0

            if len(episode_reward_buffer) == 10:
                avg10_reward = np.mean(episode_reward_buffer)
                writer.add_scalar("Reward/Avg10", avg10_reward, global_step)
                pbar.write(f"Avg10 Reward: {avg10_reward:.2f}")
                if avg10_reward > best_avg_reward:
                    best_avg_reward = avg10_reward
                    models = {
                        "mgr_actor": mgr_actor,
                        "mgr_critic": mgr_critic,
                        "wrk_actor": wrk_actor,
                        "wrk_critic": wrk_critic,
                    }
                    optimizers = {
                        "mgr_actor_opt": mgr_actor_opt,
                        "mgr_critic_opt": mgr_critic_opt,
                        "wrk_actor_opt": wrk_actor_opt,
                        "wrk_critic_opt": wrk_critic_opt,
                    }

                    save_checkpointMA(
                        global_step,models=models,optimizers=optimizers,save_dir=ckpt_dir,is_best=True
                        )             


        if global_step >= algo_cfg["warmup_steps"] and global_step % algo_cfg["update_frequency"] == 0:
            if len(mgr_buffer) >= algo_cfg["batch_size"]:
                (meta_mb, glob_mb, opt_mb, mask_mb, act_mb, rew_mb, done_mb,
                    next_meta_mb, next_glob_mb, next_opt_mb, next_mask_mb) = mgr_buffer.sample(algo_cfg["batch_size"])
                
                meta_mb       = meta_mb.to(device)
                glob_mb       = glob_mb.to(device)
                opt_mb        = opt_mb.to(device)
                mask_mb       = mask_mb.to(device)
                act_mb        = act_mb.long().to(device)                       # [B]
                rew_mb        = rew_mb.to(device)          # [B,1]
                done_mb       = done_mb.to(device)         # [B,1]
                next_meta_mb  = next_meta_mb.to(device)
                next_glob_mb  = next_glob_mb.to(device)
                next_opt_mb   = next_opt_mb.to(device)
                next_mask_mb  = next_mask_mb.to(device)


                with torch.no_grad():
                    next_logits_m = mgr_actor(next_meta_mb, next_glob_mb, next_opt_mb, next_mask_mb) # [B, act_dim]
                    next_probs_m  = F.softmax(next_logits_m.masked_fill(next_mask_mb, -1e9), dim=-1)
                    next_logp_m   = F.log_softmax(next_logits_m.masked_fill(next_mask_mb, -1e9), dim=-1)
                    q1_t_m, q2_t_m  = mgr_target_critic.forward_q_values(next_meta_mb, next_glob_mb, next_opt_mb, next_mask_mb)
                    q_t_min_m     = torch.min(q1_t_m, q2_t_m)
                    v_next_m      = (next_probs_m * (q_t_min_m - algo_cfg["alpha"] * next_logp_m)).sum(dim=-1, keepdim=True)
                    q_target_m    = rew_mb + algo_cfg["gamma"] * (1 - done_mb) * v_next_m   
                    q_target_m = q_target_m.squeeze(1) 

                # critic loss
                q1_pred_m, q2_pred_m = mgr_critic.q_for_action(meta_mb, glob_mb, opt_mb, act_mb, mask_mb)              # [B], [B]
                q1_loss_m = F.mse_loss(q1_pred_m, q_target_m)           
                q2_loss_m = F.mse_loss(q2_pred_m, q_target_m)
                critic_loss_m =  0.5 * (q1_loss_m + q2_loss_m)
                mgr_critic_opt.zero_grad()
                critic_loss_m.backward()
                mgr_critic_opt.step()

                # actor loss
                if global_step % algo_cfg["policy_update_frequency"] == 0:
                    logits_m = mgr_actor(meta_mb, glob_mb, opt_mb, mask_mb)  
                    probs_m = F.softmax(logits_m.masked_fill(mask_mb, -1e9), dim=-1)                         
                    logp_m   = F.log_softmax(logits_m.masked_fill(mask_mb, -1e9), dim=-1)
                    q1_a_m, q2_a_m = mgr_critic.forward_q_values(meta_mb, glob_mb, opt_mb, mask_mb)
                    q_a_min_m    = torch.min(q1_a_m, q2_a_m)
                    actor_loss_m = (probs_m * (algo_cfg["alpha"] * logp_m - q_a_min_m.detach())).sum(dim=-1).mean()

                    mgr_actor_opt.zero_grad()
                    actor_loss_m.backward()
                    mgr_actor_opt.step()

                    # ---- soft update target ----
                    for p, tp in zip(mgr_critic.parameters(), mgr_target_critic.parameters()):
                        tp.data.mul_(1 - algo_cfg["tau"]).add_(algo_cfg["tau"] * p.data)

            # --------------  Worker SAC Update  -------------- #
            if len(wrk_buffer) >= algo_cfg["batch_size"]:
                (meta_wb, local_wb, glob_wb, act_wb, rew_wb, done_wb,
                next_meta_wb, next_local_wb, next_glob_wb) = \
                wrk_buffer.sample(algo_cfg["batch_size"])

                meta_wb      = meta_wb.to(device)
                local_wb     = local_wb.to(device)
                glob_wb      = glob_wb.to(device)
                act_wb       = act_wb.long().to(device)                       # [B]
                rew_wb       = rew_wb.to(device)          # [B,1]
                done_wb      = done_wb.to(device)         # [B,1]
                next_meta_wb = next_meta_wb.to(device)
                next_local_wb= next_local_wb.to(device)
                next_glob_wb = next_glob_wb.to(device)

                with torch.no_grad():
                    next_logits_w = wrk_actor(next_meta_wb, next_local_wb, next_glob_wb)          # [B,2]
                    next_probs_w  = F.softmax(next_logits_w, dim=-1)
                    next_logp_w   = F.log_softmax(next_logits_w, dim=-1)
                    q1_t_w, q2_t_w= wrk_target_critic.forward_q_values(next_meta_wb, next_local_wb, next_glob_wb)
                    q_t_min_w     = torch.min(q1_t_w, q2_t_w)
                    v_next_w      = (next_probs_w * (q_t_min_w - algo_cfg["alpha"] * next_logp_w)).sum(dim=-1, keepdim=True)
                    q_target_w    = rew_wb + algo_cfg["gamma"] * (1 - done_wb) * v_next_w
                    q_target_w = q_target_w.squeeze(1)

                q1_pred_w, q2_pred_w = wrk_critic.q_for_action(meta_wb, local_wb, glob_wb, act_wb)
                critic_loss_w = 0.5 * (F.mse_loss(q1_pred_w, q_target_w) +
                F.mse_loss(q2_pred_w, q_target_w))

                wrk_critic_opt.zero_grad()
                critic_loss_w.backward()
                wrk_critic_opt.step()

                if global_step % algo_cfg["policy_update_frequency"] == 0:
                    logits_w = wrk_actor(meta_wb, local_wb, glob_wb)
                    probs_w  = F.softmax(logits_w, dim=-1)
                    logp_w   = F.log_softmax(logits_w, dim=-1)
                    q1_a_w, q2_a_w = wrk_critic.forward_q_values(meta_wb, local_wb, glob_wb)
                    q_min_w  = torch.min(q1_a_w, q2_a_w)
                    actor_loss_w = (probs_w * (algo_cfg["alpha"] * logp_w - q_min_w.detach())).sum(dim=-1).mean()

                    wrk_actor_opt.zero_grad()
                    actor_loss_w.backward()
                    wrk_actor_opt.step()

                    for p, tp in zip(wrk_critic.parameters(), wrk_target_critic.parameters()):
                        tp.data.mul_(1 - algo_cfg["tau"]).add_(algo_cfg["tau"] * p.data)
        # if global_step < 5:  
            # print(f"step {global_step} raw_reward = {global_reward} norm_reward = {norm_reward}")
        if global_step % algo_cfg["log_interval"] == 0:
            if critic_loss_m is not None and actor_loss_m is not None:
                writer.add_scalar("Manager/Loss_Q",      critic_loss_m.item(),      global_step)
                writer.add_scalar("Manager/Loss_Policy", actor_loss_m.item(), global_step)

            # Worker losses
            if critic_loss_w is not None and actor_loss_w is not None:
                writer.add_scalar("Worker/Loss_Q",      critic_loss_w.item(),      global_step)
                writer.add_scalar("Worker/Loss_Policy", actor_loss_w.item(), global_step)

            if logger and all(v is not None for v in
                  (critic_loss_m, critic_loss_w, actor_loss_m, actor_loss_w)):
                logger.info(
                    f"[{global_step}] "
                    f"MgrQ={critic_loss_m:.4f}  WkrQ={critic_loss_w:.4f}  "
                    f"MgrP={actor_loss_m:.4f}  WkrP={actor_loss_w:.4f}"
                )
                pbar.set_description(
                    f"M_Q {critic_loss_m:.3f}  W_Q {critic_loss_w:.3f}  "
                    f"M_P {actor_loss_m:.3f}  W_P {actor_loss_w:.3f}"
                )
        if global_step > 0 and global_step % algo_cfg["save_interval"] == 0:
            models = {
                "mgr_actor": mgr_actor, "mgr_critic": mgr_critic,
                "wrk_actor": wrk_actor, "wrk_critic": wrk_critic
            }
            optimizers = {
                "mgr_actor_opt": mgr_actor_opt, "mgr_critic_opt": mgr_critic_opt,
                "wrk_actor_opt": wrk_actor_opt, "wrk_critic_opt": wrk_critic_opt
            }

            save_checkpointMA(global_step, models, optimizers, ckpt_dir, is_best=False)
    
            if logger:
                logger.info(f"[{global_step}] Periodic checkpoint saved.")
                pbar.write(f"saved checkpoint at step {global_step}")

        if global_step > 0 and global_step % algo_cfg["eval_frequency"] == 0:
            if logger:
                logger.info(f"Eval begins at step {global_step}")

                mgr_actor.eval();  wrk_actor.eval()

                total_eval_reward = 0.0
                for ep in range(algo_cfg["eval_episodes"]):
                    obs_dict, _ = eval_env.reset(seed=eval_seed + ep)
                    ep_ret, done_flag = 0.0, False

                    while not done_flag:
                        meta_m, opt_m, mask_m, glob_m= [], [], [], []

                        for dc in eval_env._dc_ids:
                            o_mgr = obs_dict[f"manager_{dc}"]
                            meta_m.append(o_mgr["obs_manager_meta_task_i"])
                            opt_m.append(o_mgr["obs_all_options_set_padded"])
                            mask_m.append(o_mgr["all_options_padding_mask"])
                            glob_m.append(o_mgr["global_context"])

                        meta_m = torch.from_numpy(np.asarray(meta_m, dtype=np.float32)).to(device)   # [N_dc, D_meta_m]
                        opt_m  = torch.from_numpy(np.asarray(opt_m,  dtype=np.float32)).to(device)   # [N_dc, max_opt, D_opt]
                        mask_m = torch.from_numpy(np.asarray(mask_m, dtype=np.bool_  )).to(device)   # [N_dc, max_opt]
                        glob_m = torch.from_numpy(np.asarray(glob_m, dtype=np.float32)).to(device)   # [N_dc, D_global]

                        with torch.no_grad():
                            act_m, _, _ = mgr_actor.sample_action(meta_m, glob_m, opt_m, mask_m)
                        mgr_act = {dc: act_m[i].item() for i, dc in enumerate(env._dc_ids)}
                        obs_after_mgr = eval_env.manager_step(mgr_act)       

        
                        meta_w, local_w, glob_w = [], [], []
                        for dc in eval_env._dc_ids:
                            w = obs_after_mgr[f"worker_{dc}"]
                            meta_w.append(w["obs_worker_meta_task_i"])
                            local_w.append(w["obs_local_dc_i_for_worker"])
                            glob_w.append(w["global_context"])

                        meta_w  = torch.from_numpy(np.asarray(meta_w ,dtype=np.float32)).to(device)
                        local_w = torch.from_numpy(np.asarray(local_w,dtype=np.float32)).to(device)
                        glob_w  = torch.from_numpy(np.asarray(glob_w ,dtype=np.float32)).to(device)

                        with torch.no_grad():
                            act_w, _, _ = wrk_actor.sample_action(meta_w, local_w, glob_w)
                        wrk_act = {dc: act_w[i].item() for i, dc in enumerate(env._dc_ids)}
                        eval_env.worker_step(wrk_act)

                        next_obs, rew_dict, dones_dict, trunc_dict, _ = eval_env.env_step() 
                        
                        ep_ret += rew_dict[next(iter(rew_dict))]
                        done_flag = dones_dict["__all__"] or trunc_dict["__all__"]
                        obs_dict = next_obs  

                    total_eval_reward += ep_ret
                    if logger:
                        logger.info(f"Eval-Ep{ep+1}/{algo_cfg['eval_episodes']}  return={ep_ret:.2f}")

                avg_ret = total_eval_reward / algo_cfg["eval_episodes"]
                writer.add_scalar("Eval/AverageReturn", avg_ret, global_step)
                pbar.write(f"eval return = {avg_ret:.2f}")

                if avg_ret > best_eval_reward:
                    best_eval_reward = avg_ret
                    models = {
                        "mgr_actor": mgr_actor, "mgr_critic": mgr_critic,
                        "wrk_actor": wrk_actor, "wrk_critic": wrk_critic
                    }
                    optims = {
                        "mgr_actor_opt": mgr_actor_opt, "mgr_critic_opt": mgr_critic_opt,
                        "wrk_actor_opt": wrk_actor_opt, "wrk_critic_opt": wrk_critic_opt
                    }
                save_checkpointMA(global_step, models, optims,
                        ckpt_dir, filename="best_eval_ckpt.pth", is_best=True)
                pbar.write(f"New BEST model saved (avg return {avg_ret:.2f})")
            mgr_actor.train();  wrk_actor.train()

    writer.close()
    if logger: logger.info("Training finished.")

if __name__ == "__main__":
    train()