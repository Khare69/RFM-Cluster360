"""
Centralized structured logging module for RFM-Cluster360.
Configures human-readable console and file logging.
"""

import logging
import os
import sys
from typing import Optional

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_INITIALIZED = False


def setup_logger(
    name: str = "rfm_cluster360",
    level: Optional[int] = None,
) -> logging.Logger:
    """
    Configures and returns a logger instance for RFM-Cluster360.

    Parameters
    ----------
    name : str, default 'rfm_cluster360'
        Logger name.
    level : int, optional
        Logging level (e.g. logging.INFO). If None, reads RFM_LOG_LEVEL or defaults to INFO.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    global _INITIALIZED
    logger = logging.getLogger(name)

    if level is None:
        env_level = os.environ.get("RFM_LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, env_level, logging.INFO)
    else:
        log_level = level

    logger.setLevel(log_level)

    if not _INITIALIZED and not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        _INITIALIZED = True

    return logger
