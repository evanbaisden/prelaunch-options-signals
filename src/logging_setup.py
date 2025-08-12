import logging
import random
import numpy as np
import pandas as pd


def setup_logging(level: str) -> logging.Logger:
    """
    Set up logging configuration with specified level.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {level.upper()}")
    
    return logger


def set_seeds(seed: int) -> None:
    """
    Set seeds for reproducibility across random, numpy, and pandas.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    pd.core.common.random_state(seed)
    
    logging.getLogger(__name__).info(f"Seeds set to {seed} for reproducibility")