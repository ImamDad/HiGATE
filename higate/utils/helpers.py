"""Helper utilities."""

import torch
import numpy as np
import random
import logging
import os
from pathlib import Path

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(log_dir: str = "logs", log_level: int = logging.INFO):
    """Setup logging configuration."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/higate.log"),
            logging.StreamHandler()
        ]
    )

def count_parameters(model: torch.nn.Module) -> int:
    """Count number of trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_predictions(predictions: dict, save_path: str):
    """Save predictions to file."""
    torch.save(predictions, save_path)

def load_predictions(load_path: str) -> dict:
    """Load predictions from file."""
    return torch.load(load_path)
