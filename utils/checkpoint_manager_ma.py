# utils/checkpoint_manager_ma.py
import os
import shutil
import torch
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

def save_checkpointMA(step: int,
                    models: Dict[str, torch.nn.Module],
                    optimizers: Dict[str, torch.optim.Optimizer] | None,
                    save_dir: str,
                    is_best: bool = False,
                    filename: str | None = None,
                    extra_info: Dict[str, Any] | None = None) -> str:
    """
    Save a checkpoint that may contain multiple networks and optimizers.

    """
    os.makedirs(save_dir, exist_ok=True)

    ckpt: Dict[str, Any] = {
        "step": step,
        "model_state_dict": {name: net.state_dict() for name, net in models.items()},
        "optimizer_state_dict": ({name: opt.state_dict() for name, opt in optimizers.items()}
                                 if optimizers else {}),
        "extra_info": extra_info or {},
    }

    # Auto-generate filename if the user did not provide one
    if filename is None:
        filename = "best_checkpoint.pth" if is_best else f"checkpoint_step_{step}.pth"
    path = os.path.join(save_dir, filename)

    torch.save(ckpt, path)
    logger.info(f"[Checkpoint] step={step} saved → {path}")

    # Keep an always-present copy named ``best.pth`` for convenience
    if is_best and filename != "best_checkpoint.pth":
        shutil.copy(path, os.path.join(save_dir, "best_checkpoint.pth"))

    return path

def load_checkpoint_data(path: str,
                         device: str | torch.device = "cpu"
                         ) -> Tuple[dict[str, Any] | None, int]:
    """
    Load the checkpoint dictionary from disk.
    """
    if not os.path.exists(path):
        logger.error(f"Checkpoint not found: {path}")
        return None, 0

    try:
        ckpt = torch.load(path, map_location=device)
        step = ckpt.get("step", 0)
        logger.info(f"[Checkpoint] loaded step={step} from {path}")
        return ckpt, step
    except Exception as e:
        logger.error(f"Failed to load checkpoint {path}: {e}")
        return None, 0
