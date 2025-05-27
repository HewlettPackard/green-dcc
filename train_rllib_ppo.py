# train_rllib_ppo.py
import os
os.environ["RAY_DEDUP_LOGS"] = "1" # Set 0 to disable deduplication

import yaml
import argparse
import datetime
import logging
import numpy as np
import pandas as pd

import ray
from ray import air, tune
# from ray.rllib.algorithms.ppo import PPOConfig # Using the new PPOConfig class
from ray.tune.registry import register_env
from ray.tune import CLIReporter

# Import your custom environment
from envs.task_scheduling_env import TaskSchedulingEnv # Adjust path if necessary
# Import make_env to be used by the registration lambda
from utils.config_loader import load_yaml # Assuming this is used by your make_env
import pandas as pd # make_env might use it
from simulation.cluster_manager import DatacenterClusterManager # make_env uses it
from rewards.predefined.composite_reward import CompositeReward # make_env uses it

# --- Environment Creator Function for RLlib ---
def env_creator(env_config_rllib):
    """
    env_config_rllib will contain parameters passed from RLlib's .environment(env_config=...)
    These include the paths to your simulation config files.
    """
    print(f"RLlib env_config received by creator: {env_config_rllib}")
    sim_cfg_path = env_config_rllib.get("sim_config_path", "configs/env/sim_config.yaml")
    dc_cfg_path = env_config_rllib.get("dc_config_path", "configs/env/datacenters.yaml")
    reward_cfg_path = env_config_rllib.get("reward_config_path", "configs/env/reward_config.yaml")

    # --- Using a simplified make_env or direct instantiation ---
    # The make_env from your train_sac.py might be too complex if it sets up loggers/writers
    # specific to that script. For RLlib, it's cleaner if the env is self-contained.

    sim_cfg_full = load_yaml(sim_cfg_path)
    sim_cfg = sim_cfg_full["simulation"] # Extract the simulation part
    dc_cfg = load_yaml(dc_cfg_path)["datacenters"]
    reward_cfg = load_yaml(reward_cfg_path)["reward"]
    
    worker_idx = env_config_rllib.worker_index
    vector_idx = env_config_rllib.vector_index
    # Create a base seed that's unique per worker and sub-environment
    initial_env_seed = env_config_rllib.get("seed", 42) + worker_idx * 1000 + vector_idx * 100
    
    print(f"Worker index: {worker_idx}, Vector index: {vector_idx}, Initial seed: {initial_env_seed}")

    # Ensure 'single_action_mode' is true for this training script's purpose
    if not sim_cfg.get("single_action_mode", False):
        print("WARNING: 'single_action_mode' is not true in sim_config.yaml. This RLlib script expects it for simplicity.")
        # sim_cfg["single_action_mode"] = True # Optionally force it

    start = pd.Timestamp(datetime.datetime(sim_cfg["year"], sim_cfg["month"], sim_cfg["init_day"],
                                           sim_cfg["init_hour"], 0, tzinfo=datetime.timezone.utc))
    end = start + datetime.timedelta(days=sim_cfg["duration_days"])

    cluster = DatacenterClusterManager(
        config_list=dc_cfg,
        simulation_year=sim_cfg["year"],
        init_day=int(sim_cfg["month"] * 30.5 + sim_cfg["init_day"]),
        init_hour=sim_cfg["init_hour"],
        strategy="manual_rl", # Must be manual_rl for agent control
        tasks_file_path=sim_cfg["workload_path"],
        shuffle_datacenter_order=sim_cfg.get("shuffle_datacenters", True),
        cloud_provider=sim_cfg["cloud_provider"],
        logger=None # RLlib workers usually handle their own logging
    )

    reward_fn_instance = CompositeReward(
        components=reward_cfg["components"],
        normalize=reward_cfg.get("normalize", False),
        freeze_stats_after_steps=reward_cfg.get("freeze_stats_after_steps", None)
    )

    # Pass the sim_cfg dictionary to TaskSchedulingEnv for single_action_mode etc.
    env = TaskSchedulingEnv(
        cluster_manager=cluster,
        start_time=start,
        end_time=end,
        reward_fn=reward_fn_instance,
        writer=None, # RLlib handles its own TensorBoard logging
        sim_config=sim_cfg, # Pass the simulation config dict
        initial_seed_for_resets = initial_env_seed # Add this to TaskSchedulingEnv constructor

    )
    print(f"GreenDCC Env Created. Obs Space: {env.observation_space}, Act Space: {env.action_space}")
    return env

def main(args):
    # --- Register Environment ---
    env_name = "GreenDCC_RLlib_Env"
    register_env(env_name, env_creator)

    # --- Load RLlib PPO Configuration from YAML ---
    with open(args.rllib_config, 'r') as f:
        rllib_full_config = yaml.safe_load(f)

    algo_specific_config_dict = rllib_full_config.get("algorithm", {})
    run_config_dict = rllib_full_config.get("run_config", {})

    # --- Create PPOConfig Object ---
    # Start with default PPO config and update with our settings
    # Note: RLlib's PPOConfig().training(), .resources() etc. are the new way.
    # For direct dict passing, ensure keys match what PPOConfig expects.
    # We are building the config dict directly from YAML for tune.Tuner

    # Ensure the 'env' key in algo_specific_config_dict uses the registered name
    algo_specific_config_dict["env"] = env_name

    # --- Configure Progress Reporter for Tune ---
    reporter = CLIReporter(max_progress_rows=10)
    reporter.add_metric_column("evaluation/env_runners/episode_return_mean")
    reporter.add_metric_column("env_runners/episode_return_mean")


    # --- Initialize Ray ---
    if ray.is_initialized():
        ray.shutdown()
    
    ray.init(
        num_cpus=args.num_cpus if args.num_cpus is not None else 32, # Example: use up to 32 CPUs
        num_gpus=args.num_gpus if args.num_gpus is not None else 0,
        include_dashboard=False if algo_specific_config_dict.get("num_env_runners") < 2 else True,
        ignore_reinit_error=True,
        logging_level=logging.ERROR if not args.ray_verbose else logging.INFO,
        # Set local_mode=True for local testing when in the config, the num_env_runners is lower than 2
        local_mode = True if algo_specific_config_dict.get("num_env_runners") < 2 else False,
    )

    # --- Setup Tune Experiment ---
    experiment_name = run_config_dict.get("name", f"GreenDCC_PPO_Experiment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    stop_conditions = run_config_dict.get("stop", {"training_iteration": 100})
    checkpoint_config_dict = run_config_dict.get("checkpoint_config", {})

    metric_to_monitor = checkpoint_config_dict.get("checkpoint_score_attribute", "evaluation/env_runners/episode_return_mean")
    metric_mode = checkpoint_config_dict.get("checkpoint_score_order", "max")
    
    tuner = tune.Tuner(
        "PPO", # Algorithm name
        param_space=algo_specific_config_dict, # Algorithm-specific config
        run_config=air.RunConfig(
            name=experiment_name,
            stop=stop_conditions,
            checkpoint_config=air.CheckpointConfig(**checkpoint_config_dict),
            verbose=run_config_dict.get("verbose", 1), # Tune verbosity
            storage_path=args.results_dir, # Where to save Tune results
            progress_reporter=reporter

        )
    )

    # --- Run Training ---
    print(f"Starting Tune experiment: {experiment_name}")
    results = tuner.fit()
    print("Training finished.")
    
    # # Get last reported results per trial
    # df = results.get_dataframe()
    # if df is not None and not df.empty:
    #     print("--- DataFrame Columns ---")
    #     # Print the column name and the value of the first row
    #     for col_name in df.columns:
    #         print(f"{col_name}: {df[col_name].iloc[0]}")
    
    #     # for col_name in df.columns:
    #     #     print(col_name)
    #     # print("\n--- First 5 Rows of DataFrame ---")
    #     print(df.head()) # Print the first 5 rows to see some data
    #     print("\n--- DataFrame Info (Data Types, Non-Null Counts) ---")
    #     df.info()
    # else:
    #     print("No data in results DataFrame (e.g., all trials failed early or no metrics reported).")

    # list_of_dfs = results.get_dataframe() # Assuming it returns a list
    # if list_of_dfs:
    #     for i, df_trial in enumerate(list_of_dfs):
    #         if df_trial is not None and not df_trial.empty:
    #             print(f"--- Columns for Trial {i} ---")
    #             for col_name in df_trial.columns:
    #                 print(col_name)
    #             print(f"\n--- First 5 Rows of Trial {i} DataFrame ---")
    #             print(df_trial.head())
    #             print("\n")
    #         else:
    #             print(f"Trial {i} resulted in an empty DataFrame.")
    # else:
    #     print("No DataFrames returned from results.")
    
       
    # --- Post-Training ---
    print("\n--- Summary of Trial Statuses ---")
    print(f"Total trials: {len(results)}")
    print(f"Number of terminated (successful) trials: {results.num_terminated}")
    print(f"Number of errored trials: {results.num_errors}")
    
    if results.errors:
        print("\n--- WARNING Details of Errored Trials ---")
        for i, trial_result in enumerate(results): # Iterating here gives individual Result objects
            if trial_result.error:
                print(f"Trial {i} (Path: {trial_result.path}):") # trial_result.trial_id might not be directly available on Result, path is good
                print(f"  Error Type: {type(trial_result.error).__name__}")
                print(f"  Error Message: {trial_result.error}")
                print(f"  Config: {trial_result.config}")
                
    best_result = results.get_best_result()
    if best_result:
        print("\n--- Best Trial Information ---")
        print(f"Best trial config: {best_result.config.get('algorithm', best_result.config)}") # Show algo config
        best_df = best_result.metrics_dataframe
        if best_df is not None and not best_df.empty:
            print(f"Best trial DataFrame: {best_df.head()}")
            # Print the column names and the first row of the best trial DataFrame
            print("\n--- Best Trial DataFrame Columns ---")
            for col_name in best_df.columns:
                print(f"{col_name}: {best_df[col_name].iloc[0]}")
            print("\n--- Best Trial DataFrame Info (Data Types, Non-Null Counts) ---")
            best_df.info()
        else:
            print("No data in best trial DataFrame (e.g., all trials failed early or no metrics reported).")
            
        print(f"Best trial final monitored validation metric ({metric_to_monitor}): {best_result.metrics['evaluation']['env_runners']['episode_return_mean']:.3f}")
        
        # Accessing the best checkpoint path:
        # The best_result.checkpoint might be None if no checkpoints were saved or if criteria weren't met
        # Checkpoints are typically stored within the trial's directory in the storage_path
        # The `best_result.path` gives the directory of the best trial.
        # The actual best checkpoint according to score is often symlinked or recorded by Tune.
        
        # A more robust way to get the best checkpoint according to the score:
        # Tune often creates a 'best_checkpoint' symlink or a file indicating it within the experiment dir.
        # However, best_result.checkpoint directly gives you the AIR Checkpoint object for the last checkpoint OF THE BEST TRIAL.
        # To get the checkpoint that specifically scored the best on the metric:
        best_trial_logdir = best_result.path
        print(f"Log directory of the best trial: {best_trial_logdir}")

        # RLlib typically saves checkpoints within the trial logdir.
        # The best_result.checkpoint object refers to the *last* checkpoint of this best trial.
        # If checkpoint_score_attribute was used, Tune also separately tracks the best checkpoint.
        # The `results.get_best_checkpoint(trial=best_result, metric=..., mode=...)` was an older API.
        # Now, you often rely on the checkpointing behavior within the trial's directory.
        # Let's print the path of the last checkpoint of the best trial,
        # which is usually the best one if checkpoint_at_end=True and it was the best overall.
        if best_result.checkpoint:
             print(f"Path to the last checkpoint of the best trial: {best_result.checkpoint.path}")
             print(f"This checkpoint might be the best if 'checkpoint_at_end=True' and it scored highest.")
        else:
             print("No checkpoint reported for the best trial by `best_result.checkpoint`.")
        
        print(f"\nTo find the absolute best checkpoint based on '{metric_to_monitor}', "
              f"you might need to inspect the trial directories within: {args.results_dir}/{experiment_name}")
        print("Tune typically creates a 'checkpoint_best' or similar based on the scoring attribute.")

    else:
        print("No best result found (e.g., all trials might have failed).")

    ray.shutdown()
    print("Ray shutdown.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GreenDCC with RLlib PPO")
    parser.add_argument(
        "--rllib-config", type=str, default="configs/rllib/ppo_config.yaml",
        help="Path to the RLlib PPO configuration YAML file."
    )
    parser.add_argument(
        "--results-dir", type=str, default="~/ray_results", # Ray's default
        help="Directory to save Tune experiment results."
    )
    parser.add_argument(
        "--num-cpus", type=int, default=None, # Let Ray auto-detect
        help="Number of CPUs to allocate for Ray. Default: Ray auto-detects."
    )
    parser.add_argument(
        "--num-gpus", type=int, default=None, # Let Ray auto-detect
        help="Number of GPUs to allocate for Ray. Default: Ray auto-detects."
    )
    parser.add_argument(
        "--ray-verbose", action="store_true",
        help="Enable more verbose Ray logging."
    )
    cli_args = parser.parse_args()
    main(cli_args)