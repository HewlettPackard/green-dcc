from pprint import pprint

from gymnasium import spaces
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.impala import IMPALAConfig
from ray.tune.logger import TBXLoggerCallback

from envs.task_scheduling_env import TaskSchedulingEnv
from train_rl_agent import make_env, parse_args
from utils.config_loader import load_yaml

class TaskSchedulingEnvRLLIB(TaskSchedulingEnv):

    def __init__(self, config = {}):
        args = parse_args()
        
        ts_env_dummy = make_env(args.sim_config, args.dc_config, args.reward_config)
        cluster_manager = ts_env_dummy.cluster_manager
        start_time = ts_env_dummy.start_time
        end_time = ts_env_dummy.end_time
        reward_fn = ts_env_dummy.reward_fn

        super().__init__(cluster_manager, start_time, end_time, reward_fn)

        algo_cfg = load_yaml(args.algo_config)["algorithm"]
        self.max_tasks = algo_cfg["max_tasks"]

        # Observation space: [4 sin/cos features, 4 task features features, 5 * num_dcs task features]
        self.obs_dim = 4 + 5 + 5 * self.num_dcs

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.max_tasks*self.obs_dim,),
            dtype=np.float32
        )
        
        self.action_space = spaces.MultiDiscrete([1 + self.num_dcs] * self.max_tasks)
        
        # For DQN I need to change to discrete action space instead of multi-discrete -> ((1 + self.num_dcs) * self.max_tasks)

    def step(self, actions):
        obs, reward, done, truncated, info = super().step(actions[:len(self.current_tasks)])
        return self._reshape_obs(obs), reward, done, truncated, info

    def _reshape_obs(self, obs):
        if obs:
            obs = np.asarray(obs, dtype=np.float32)
            N, B = obs.shape
            obs = np.pad(obs, ((0, self.max_tasks-N), (0, 0)), mode='constant', constant_values=0)
            obs = obs.reshape(self.max_tasks*B)
        else:
            obs = np.zeros((self.max_tasks*self.obs_dim,), dtype=np.float32)
        return obs
    
    def reset(self, seed=None, options=None):
        obs, _ = super().reset(seed, options)
        return self._reshape_obs(obs), {}


if __name__ == '__main__':
    # Test the environment
    # env = TaskSchedulingEnvRLLIB({})
    # obs, _ = env.reset()
    # done = False
    # step = 0
    # while not done:
    #     action = env.action_space.sample()
    #     obs, reward, done, truncated, info = env.step(action)
    #     step += 1
    #     pprint(info['datacenter_infos']['DC1'])
    # exit()

    # Set up training 
    ray.init(local_mode=True)

    config = (
        APPOConfig()
        .environment(TaskSchedulingEnvRLLIB)
        .framework("torch")
        .training(
            num_epochs=1,
            train_batch_size=672,
            )
        .env_runners(
            num_env_runners=0,
            rollout_fragment_length=1
        )
        .learners(
            num_gpus_per_learner=0
        )
        .checkpointing()
    )

    # config = (
    #     IMPALAConfig()
    #     .environment(TaskSchedulingEnvRLLIB)
    #     .framework("torch")
    #     .training(
    #         num_epochs=1,
    #         train_batch_size=672,
    #         # minibatch_size=128
    #     )
    #     .env_runners(
    #         num_env_runners=0,
    #         rollout_fragment_length=1
    #     )
    #     .learners(
    #         num_gpus_per_learner=0
    #     )
    # )

    # algo = config.build_algo()
    # for _ in range(100):
    #     results = algo.train()
    #     pprint(results['env_runners']['episode_return_mean'])
    # exit()

    RESULTS_DIR = "/lustre/guillant/new_green-dcc/results"
    NAME = "test"

    tuner = tune.Tuner(
        "APPO",
        tune_config=tune.TuneConfig(
            metric="env_runners/episode_return_mean",
            mode='max'
        ),
        run_config=tune.RunConfig(
            name=NAME,
            stop={"training_iteration": 100},
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=1,
                ),
            storage_path=RESULTS_DIR,
        ),
        param_space=config
    )

    tuner.fit()