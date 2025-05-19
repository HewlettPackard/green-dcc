.. _train:

Train
=====

To train and evaluate the Top-level RL agent, run the `train_rl_agent.py` script with the desired configurations:

.. code-block:: bash

   python train_rl_agent.py \
     --sim-config    configs/env/sim_config.yaml \
     --reward-config configs/env/reward_config.yaml \
     --dc-config     configs/env/datacenters.yaml \
     --algo-config   configs/env/algorithm_config.yaml \
     [--tag <run_tag>] \
     [--seed <random_seed>] \
     [--enable-logger true|false]


Command-Line Arguments
----------------------

The script accepts the following options:

.. option:: --sim-config SIM_CONFIG

   Path to the simulation configuration YAML file.  
   **Default:** ``configs/env/sim_config.yaml``.

.. option:: --reward-config REWARD_CONFIG

   Path to the reward configuration YAML file.  
   **Default:** ``configs/env/reward_config.yaml``.

.. option:: --dc-config DC_CONFIG

   Path to the datacenter configuration YAML file.  
   **Default:** ``configs/env/datacenters.yaml``.

.. option:: --algo-config ALGO_CONFIG

   Path to the reinforcement-learning algorithm configuration YAML file.  
   **Default:** ``configs/env/algorithm_config.yaml``.

.. option:: --tag TAG

   Optional run tag to distinguish logs and checkpoints.  
   **Default:** (empty string).

.. option:: --seed SEED

   Integer random seed for environment and training.  
   **Default:** ``42``.

.. option:: --enable-logger {yes,true,t,1}/{no,false,f,0}

   Whether to enable debug-level logger output.  
   **Default:** ``True``.


Training Script
---------------

Below is the full contents of **train_rl_agent.py**:

.. code-block:: python

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
       parser = argparse.ArgumentParser(description="SustainCluster Training")
       parser.add_argument(
           "--sim-config",
           type=str,
           default="configs/env/sim_config.yaml"
       )
       parser.add_argument(
           "--reward-config",
           type=str,
           default="configs/env/reward_config.yaml"
       )
       parser.add_argument(
           "--dc-config",
           type=str,
           default="configs/env/datacenters.yaml"
       )
       parser.add_argument(
           "--algo-config",
           type=str,
           default="configs/env/algorithm_config.yaml"
       )
       parser.add_argument(
           "--tag",
           type=str,
           default="",
           help="Optional run tag"
       )
       parser.add_argument(
           "--seed",
           type=int,
           default=42
       )
       parser.add_argument(
           "--enable-logger",
           type=str2bool,
           default=True,
           help="Enable logger"
       )
       return parser.parse_args()


   def make_env(sim_cfg_path, dc_cfg_path, reward_cfg_path, writer=None, logger=None):
       import pandas as pd
       from simulation.cluster_manager import DatacenterClusterManager

       sim_cfg    = load_yaml(sim_cfg_path)["simulation"]
       dc_cfg     = load_yaml(dc_cfg_path)["datacenters"]
       reward_cfg = load_yaml(reward_cfg_path)["reward"]

       start = pd.Timestamp(
           datetime.datetime(
               sim_cfg["year"],
               sim_cfg["month"],
               sim_cfg["init_day"],
               sim_cfg["init_hour"],
               tzinfo=datetime.timezone.utc
           )
       )
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
       args      = parse_args()
       timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
       run_id    = f"{args.tag}_{timestamp}" if args.tag else timestamp

       log_dir  = f"logs/train_{run_id}"
       tb_dir   = f"runs/train_{run_id}"
       ckpt_dir = f"checkpoints/train_{run_id}"
       os.makedirs(ckpt_dir, exist_ok=True)

       writer = SummaryWriter(log_dir=tb_dir)
       print(f"Enable logger: {args.enable_logger}")
       logger = setup_logger(log_dir, enable_logger=args.enable_logger)

       algo_cfg = load_yaml(args.algo_config)["algorithm"]
       if algo_cfg["device"] == "auto":
           DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       else:
           DEVICE = torch.device(algo_cfg["device"])

       env = make_env(
           args.sim_config,
           args.dc_config,
           args.reward_config,
           writer,
           logger
       )

       obs, _ = env.reset(seed=args.seed)
       while len(obs) == 0:
           obs, _, done, _, _ = env.step([])
           if done:
               obs, _ = env.reset(seed=args.seed)

       obs_dim = len(obs[0])
       act_dim = env.num_dcs + 1

       actor         = ActorNet(obs_dim, act_dim,
                                hidden_dim=algo_cfg["hidden_dim"]).to(DEVICE)
       critic        = CriticNet(obs_dim, act_dim,
                                hidden_dim=algo_cfg["hidden_dim"]).to(DEVICE)
       target_critic = CriticNet(obs_dim, act_dim,
                                hidden_dim=algo_cfg["hidden_dim"]).to(DEVICE)
       target_critic.load_state_dict(critic.state_dict())

       actor_opt  = torch.optim.Adam(
           actor.parameters(),
           lr=float(algo_cfg["actor_learning_rate"])
       )
       critic_opt = torch.optim.Adam(
           critic.parameters(),
           lr=float(algo_cfg["critic_learning_rate"])
       )

       buffer = FastReplayBuffer(
           capacity=algo_cfg["replay_buffer_size"],
           max_tasks=algo_cfg["max_tasks"],
           obs_dim=obs_dim
       )

       reward_stats          = RunningStats()
       episode_reward        = 0
       episode_steps         = 0
       episode_reward_buffer = deque(maxlen=10)
       best_avg_reward       = float("-inf")
       q_loss = policy_loss = None

       pbar = trange(algo_cfg["total_steps"])

       for global_step in pbar:
           obs_tensor = torch.FloatTensor(obs).to(DEVICE)
           if not obs:
               actions = []
           elif global_step < algo_cfg["warmup_steps"]:
               actions = [np.random.randint(act_dim) for _ in obs]
           else:
               with torch.no_grad():
                   logits = actor(obs_tensor)
                   probs  = F.softmax(logits, dim=-1)
                   dist   = torch.distributions.Categorical(probs)
                   actions= dist.sample().cpu().tolist()
                   assert all(0 <= a < act_dim for a in actions), \
                          f"Invalid action: {actions}"
                   writer.add_histogram("Actor/logits", logits, global_step)
                   writer.add_histogram("Actor/probs",  probs,  global_step)

           next_obs, reward, done, truncated, _ = env.step(actions)
           reward_stats.update(reward)
           normalized_reward = reward_stats.normalize(reward)

           if actions:
               buffer.add(obs, actions, normalized_reward,
                          next_obs, done or truncated)

           obs             = next_obs
           episode_reward += reward
           episode_steps  += 1

           if done or truncated:
               avg = episode_reward / episode_steps
               episode_reward_buffer.append(avg)
               if logger:
                   logger.info(f"[Episode End] total_reward={avg:.2f}")
               writer.add_scalar("Reward/Episode", avg, global_step)
               pbar.write(f"Episode reward: {avg:.2f} (steps: {episode_steps})")
               obs, _ = env.reset(seed=args.seed+global_step)
               episode_reward = 0
               episode_steps  = 0
               if len(episode_reward_buffer) == 10:
                   avg10 = np.mean(episode_reward_buffer)
                   writer.add_scalar("Reward/Avg10", avg10, global_step)
                   pbar.write(f"Avg reward: {avg10:.2f}")
                   if avg10 > best_avg_reward:
                       best_avg_reward = avg10
                       save_checkpoint(
                           global_step, actor, critic,
                           actor_opt, critic_opt,
                           ckpt_dir, best=True
                       )
                       pbar.write(
                         f"[BEST] Saved checkpoint at step {global_step} "
                         f"(avg10 reward={avg10:.2f})"
                       )

           # RL updates (Q- and policy-loss, backward, optimizer steps) omitted for brevity

           if global_step % algo_cfg["save_interval"] == 0 and global_step > 0:
               save_checkpoint(global_step, actor, critic,
                               actor_opt, critic_opt, ckpt_dir)
               pbar.write(f"Saved checkpoint at step {global_step}")

       writer.close()


   if __name__ == "__main__":
       train()
