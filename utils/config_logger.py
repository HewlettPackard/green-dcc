import os
import logging
from datetime import datetime

def setup_logger(log_dir: str, enable_logger: bool):
    if not enable_logger:
        return None

    os.makedirs(log_dir, exist_ok=True)

    # Format: train_YYYYMMDD_HHMMSS.log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"train_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)

    # Clear previous handlers (e.g., in Jupyter)
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(console_handler)

    return logger
