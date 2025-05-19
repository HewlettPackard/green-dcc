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


def load_checkpoint_data( # Renamed for clarity, or keep old name and adjust call
    path,
    device="cpu",
    # Remove actor, critic, opts from here if they are not used for loading decisions
):
    """Loads checkpoint data dictionary. Does NOT load into model instances."""
    if not os.path.exists(path):
        if logger: logger.error(f"Checkpoint file not found: {path}")
        return None, 0 # Indicate failure

    try:
        checkpoint = torch.load(path, map_location=device)
        loaded_step = checkpoint.get("step", 0)

        # We are returning the whole dictionary.
        # The caller will be responsible for extracting state_dicts and extra_info
        # and then loading them into appropriately instantiated models.

        if logger: logger.info(f"Checkpoint data loaded successfully from step {loaded_step} at {path}.")
        return checkpoint, loaded_step

    except Exception as e:
        if logger: logger.error(f"Error loading checkpoint data from {path}: {e}")
        return None, 0