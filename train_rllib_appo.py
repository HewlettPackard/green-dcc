# train_rllib_appo.py
import os
# --- Set RAY_DEDUP_LOGS before importing Ray if needed for debugging ---
# os.environ["RAY_DEDUP_LOGS"] = "0" # "0" to disable, "1" or unset to enable

import yaml
import argparse
import datetime
import logging
import pandas as pd # For Timestamp and Timedelta

import ray
from ray import air, tune
from ray.tune.registry import register_env
from ray.tune import CLIReporter

# Import your custom environment and helpers
from envs.task_scheduling_env import TaskSchedulingEnv # Adjust path if necessary
from utils.config_loader import load_yaml
from simulation.cluster_manager import DatacenterClusterManager
from rewards.predefined.composite_reward import CompositeReward
from train_rllib_ppo import env_creator


def main(args):
    env_name = "GreenDCC_RLlib_Env"
    register_env(env_name, env_creator)

    with open(args.rllib_config, 'r') as f:
        rllib_full_config = yaml.safe_load(f)
    appo_param_space = rllib_full_config.get("algorithm", {})
    run_params = rllib_full_config.get("run_config", {})

    appo_param_space["env"] = env_name # Ensure registered env name is used

    # --- Configure Progress Reporter ---
    reporter = CLIReporter(max_progress_rows=10)
    # Add metrics you expect APPO to log for training and evaluation
    # These might differ slightly from PPO's default naming for some internal stats
    # Common ones:
    reporter.add_metric_column("evaluation/episode_reward_mean") # From evaluation rollouts
    reporter.add_metric_column("info/learner/default_policy/learner_stats/vf_loss") # Example learner stat
    reporter.add_metric_column("info/learner/default_policy/learner_stats/policy_loss")


    if not ray.is_initialized():
        ray.init(
            num_cpus=args.num_cpus if args.num_cpus is not None else 32, # Example: use up to 32 CPUs
            num_gpus=args.num_gpus if args.num_gpus is not None else 0,
            include_dashboard=False if appo_param_space.get("num_env_runners") < 2 else True,
            ignore_reinit_error=True,
            logging_level=logging.ERROR if not args.ray_verbose else logging.INFO,
            # Set local_mode=True for local testing when in the config, the num_env_runners is lower than 2
            local_mode = True if appo_param_space.get("num_env_runners") < 2 else False,
        )

    experiment_name = run_params.get("name", f"GreenDCC_APPO_Experiment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    stop_conditions = run_params.get("stop", {"training_iteration": 100})
    checkpoint_config_from_yaml = run_params.get("checkpoint_config", {})
    
    metric_to_monitor = checkpoint_config_from_yaml.get("checkpoint_score_attribute", "evaluation/episode_return_mean")
    metric_mode = checkpoint_config_from_yaml.get("checkpoint_score_order", "max")

    air_checkpoint_config = air.CheckpointConfig(
        num_to_keep=checkpoint_config_from_yaml.get("num_to_keep", None),
        checkpoint_frequency=checkpoint_config_from_yaml.get("checkpoint_frequency", 0),
        checkpoint_at_end=checkpoint_config_from_yaml.get("checkpoint_at_end", True),
        checkpoint_score_attribute=metric_to_monitor if metric_to_monitor else None, # Pass None if not set
        checkpoint_score_order=metric_mode if metric_to_monitor else None
    )

    tuner = tune.Tuner(
        "APPO", # <--- Changed algorithm name
        param_space=appo_param_space,
        run_config=air.RunConfig(
            name=experiment_name,
            stop=stop_conditions,
            checkpoint_config=air_checkpoint_config,
            verbose=run_params.get("verbose", 1),
            storage_path=os.path.expanduser(args.results_dir),
            progress_reporter=reporter
        )
    )

    print(f"Starting Tune experiment: {experiment_name} with APPO config")
    results = tuner.fit()
    print("Training finished.")

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
    parser = argparse.ArgumentParser(description="Train GreenDCC with RLlib APPO")
    parser.add_argument(
        "--rllib-config", type=str, default="configs/rllib/appo_config.yaml", # Point to APPO config
        help="Path to the RLlib APPO configuration YAML file."
    )
    parser.add_argument("--results-dir", type=str, default="~/ray_results") # Separate results dir
    parser.add_argument("--num-cpus", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--ray-verbose", action="store_true")
    cli_args = parser.parse_args()
    main(cli_args)