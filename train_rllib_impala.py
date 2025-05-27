# train_rllib_impala.py
import os
# os.environ["RAY_DEDUP_LOGS"] = "0" # Uncomment for debugging worker logs

import yaml
import argparse
import datetime
import logging
import pandas as pd

import ray
from ray import air, tune
from ray.tune.registry import register_env
from ray.tune import CLIReporter

# Import from your PPO training script that contains env_creator
# Ensure train_rllib_ppo.py is in Python's path or same directory
try:
    from train_rllib_ppo import env_creator
except ImportError:
    print("ERROR: Could not import env_creator from train_rllib_ppo.py.")
    print("Ensure train_rllib_ppo.py is in the same directory or accessible in PYTHONPATH.")
    exit()


def main(args):
    env_name = "GreenDCC_RLlib_Env" # Must match what env_creator registers if not re-registering
    # If env_creator is imported, it might have already registered.
    # To be safe, or if running standalone, register it here.
    try:
        register_env(env_name, env_creator)
        print(f"Environment '{env_name}' registered with env_creator.")
    except Exception as e:
        print(f"Note: Environment '{env_name}' might have already been registered: {e}")


    with open(args.rllib_config, 'r') as f:
        rllib_full_config = yaml.safe_load(f)
    impala_param_space = rllib_full_config.get("algorithm", {})
    run_params = rllib_full_config.get("run_config", {})

    impala_param_space["env"] = env_name

    # --- Configure Progress Reporter ---
    reporter = CLIReporter(max_progress_rows=10)
    reporter.add_metric_column("evaluation/episode_reward_mean") # Evaluation reward
    # IMPALA might have different learner stat paths, e.g.:
    reporter.add_metric_column("info/learner/learner_stats/vf_loss") # Check exact path
    reporter.add_metric_column("info/learner/learner_stats/policy_loss")


    if not ray.is_initialized():
        ray.init(
            num_cpus=args.num_cpus,
            num_gpus=args.num_gpus,
            include_dashboard=True, # Dashboard can be helpful
            logging_level=logging.ERROR if not args.ray_verbose else logging.INFO,
            local_mode = True if impala_param_space.get("num_workers", 1) < 2 and impala_param_space.get("num_learner_workers", 0) == 0 else False,
        )

    experiment_name = run_params.get("name", f"GreenDCC_IMPALA_Experiment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    stop_conditions = run_params.get("stop", {"training_iteration": 100})
    checkpoint_config_from_yaml = run_params.get("checkpoint_config", {})
    
    metric_to_monitor = checkpoint_config_from_yaml.get("checkpoint_score_attribute", "evaluation/episode_reward_mean") # VERIFY THIS KEY
    metric_mode = checkpoint_config_from_yaml.get("checkpoint_score_order", "max")

    air_checkpoint_config = air.CheckpointConfig(
        num_to_keep=checkpoint_config_from_yaml.get("num_to_keep", None),
        checkpoint_frequency=checkpoint_config_from_yaml.get("checkpoint_frequency", 0),
        checkpoint_at_end=checkpoint_config_from_yaml.get("checkpoint_at_end", True),
        checkpoint_score_attribute=metric_to_monitor if metric_to_monitor else None,
        checkpoint_score_order=metric_mode if metric_to_monitor else None
    )

    tuner = tune.Tuner(
        "IMPALA", # <--- Changed algorithm name
        param_space=impala_param_space,
        run_config=air.RunConfig(
            name=experiment_name,
            stop=stop_conditions,
            checkpoint_config=air_checkpoint_config,
            verbose=run_params.get("verbose", 1),
            storage_path=os.path.expanduser(args.results_dir),
            progress_reporter=reporter
        )
    )

    print(f"Starting Tune experiment: {experiment_name} with IMPALA config")
    results = tuner.fit()
    print("Training finished.")

    # --- Post-Training (Identical to your PPO/APPO script's post-training section) ---
    print("\n--- Summary of Trial Statuses ---")
    # ... (your existing logic to print dataframe columns, best trial info, etc.) ...
    print(f"Total trials: {len(results)}")
    print(f"Number of terminated (successful) trials: {results.num_terminated}")
    print(f"Number of errored trials: {results.num_errors}")

    # ... (rest of your post-training analysis from previous script) ...
    try:
        best_result = results.get_best_result(metric=metric_to_monitor, mode=metric_mode)
        if best_result:
            print("\n--- Best Trial Information ---")
            print(f"Best trial logdir: {best_result.path}")
            print(f"Best trial config: {best_result.config.get('algorithm', best_result.config)}")
            
            # Safely access nested metric
            best_metric_value = float('-inf' if metric_mode == 'max' else float('inf'))
            current_dict = best_result.metrics
            try:
                for key_part in metric_to_monitor.split('/'):
                    current_dict = current_dict[key_part]
                best_metric_value = current_dict
            except KeyError:
                print(f"Warning: Metric key '{metric_to_monitor}' not found in best_result.metrics dict.")
            
            print(f"Best trial final monitored validation metric ({metric_to_monitor}): {best_metric_value}")

            if best_result.checkpoint:
                 print(f"Path to the last checkpoint of the best trial: {best_result.checkpoint.path}")
            else:
                 print("No checkpoint reported for the best trial by `best_result.checkpoint`.")
        else:
            print(f"No best result found for metric '{metric_to_monitor}'.")
    except Exception as e:
        print(f"Error retrieving best result: {e}")


    ray.shutdown()
    print("Ray shutdown.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GreenDCC with RLlib IMPALA")
    parser.add_argument(
        "--rllib-config", type=str, default="configs/rllib/impala_config.yaml", # Point to IMPALA config
        help="Path to the RLlib IMPALA configuration YAML file."
    )
    parser.add_argument("--results-dir", type=str, default="~/ray_results") # Separate results dir
    parser.add_argument("--num-cpus", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--ray-verbose", action="store_true")
    cli_args = parser.parse_args()
    main(cli_args)