# logging_config.py

import logging
from pathlib import Path

# Define Project Root, assuming this file is src/transcriptanalysis/logging_config.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def setup_logging(
    enable_logging: bool,
    logging_level_str: str = "DEBUG",
    log_to_file: bool = True,
    log_file_path: str = "logs/application.log"
) -> None:
    """
    Centralized logging configuration function.

    Args:
        enable_logging (bool): Whether to enable logging or set to CRITICAL only.
        logging_level_str (str): Logging level as a string, e.g. 'DEBUG' or 'INFO'.
        log_to_file (bool): If True, logs will also be written to a file.
        log_file_path (str): Relative path for the log file (e.g., 'logs/application.log').
    """
    if enable_logging:
        # Convert string level to actual logging level (default to DEBUG if invalid)
        logging_level = getattr(logging, logging_level_str.upper(), logging.DEBUG)

        handlers = [logging.StreamHandler()]
        if log_to_file:
            # log_file_path is the relative path like "logs/application.log"
            log_file_abs_path = PROJECT_ROOT / log_file_path 
            log_file_abs_path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_file_abs_path))

        logging.basicConfig(
            level=logging_level,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=handlers
        )
    else:
        # If logging is disabled, set level to CRITICAL so minimal logs appear
        logging.basicConfig(level=logging.CRITICAL)
