# utils/checkpoint_manager.py
import os
import torch
import logging # Optional: for logging during load

logger = logging.getLogger(__name__) # Optional: use logging

def save_checkpoint(
    step,
    actor,
    critic,
    actor_opt,
    critic_opt,
    save_dir,
    is_best=False,
    filename=None, # Added optional filename override
    **kwargs # Accept extra keyword arguments (like running stats)
):
    """Saves model and optimizer states, plus optional extra data."""
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        "step": step,
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "actor_optimizer_state_dict": actor_opt.state_dict(),
        "critic_optimizer_state_dict": critic_opt.state_dict(),
    }

    # Add any extra keyword arguments provided (e.g., running stats states)
    checkpoint.update(kwargs)

    # Determine filename
    if filename is None:
        filename = "best_checkpoint.pth" if is_best else f"checkpoint_step_{step}.pth"

    path = os.path.join(save_dir, filename)
    try:
        torch.save(checkpoint, path)
        print(f"Checkpoint saved at step {step} -> {path}") # Keep simple print for progress
    except Exception as e:
        print(f"Error saving checkpoint to {path}: {e}")


def load_checkpoint(
    path,
    actor,
    critic,
    actor_opt=None,
    critic_opt=None,
    device="cpu",
    # Add args to potentially receive running stats objects
    reward_stats=None,
    critic_obs_stats=None
):
    """Loads model and optimizer states, and optionally running stats states."""
    if not os.path.exists(path):
        logger.error(f"Checkpoint file not found: {path}")
        return 0 # Return step 0 or raise error

    try:
        checkpoint = torch.load(path, map_location=device)

        # Load core components
        actor.load_state_dict(checkpoint["actor_state_dict"])
        critic.load_state_dict(checkpoint["critic_state_dict"])
        logger.info(f"Loaded actor and critic state dicts from {path}")

        # Load optimizers if provided
        if actor_opt and "actor_optimizer_state_dict" in checkpoint:
            actor_opt.load_state_dict(checkpoint["actor_optimizer_state_dict"])
            logger.info("Loaded actor optimizer state dict.")
        if critic_opt and "critic_optimizer_state_dict" in checkpoint:
            critic_opt.load_state_dict(checkpoint["critic_optimizer_state_dict"])
            logger.info("Loaded critic optimizer state dict.")

        # Load running stats if provided and present in checkpoint
        if reward_stats and "reward_stats" in checkpoint:
            try:
                reward_stats.set_state(checkpoint["reward_stats"])
                logger.info("Loaded reward running stats.")
            except Exception as e:
                logger.warning(f"Could not load reward_stats: {e}. Stats might be reset.")
        elif reward_stats:
             logger.warning("reward_stats object provided, but no reward_stats found in checkpoint.")


        if critic_obs_stats and "critic_obs_stats" in checkpoint:
            try:
                critic_obs_stats.set_state(checkpoint["critic_obs_stats"])
                logger.info("Loaded critic observation running stats.")
            except Exception as e:
                 logger.warning(f"Could not load critic_obs_stats: {e}. Stats might be reset.")
        elif critic_obs_stats:
             logger.warning("critic_obs_stats object provided, but no critic_obs_stats found in checkpoint.")


        loaded_step = checkpoint.get("step", 0)
        logger.info(f"Checkpoint loaded successfully from step {loaded_step}.")
        return loaded_step

    except Exception as e:
        logger.error(f"Error loading checkpoint from {path}: {e}")
        return 0 # Return step 0 or raise error