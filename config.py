
import numpy as np
import torch
from pathlib import Path
import logging
from typing import Tuple, List, Dict
import json

logger = logging.getLogger(__name__)

class Config:
    """Configuration class for HiGATE matching research paper specifications"""
    
    def __init__(self):
        # Hardware configuration
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.NUM_WORKERS = min(4, torch.multiprocessing.cpu_count() // 2)
        
        # Data paths - Update these according to your setup
        self.DATA_ROOT = Path("F:/PanNuke")  # Change this path
        self.TRAIN_FOLD = self.DATA_ROOT / "fold1"  # As per paper
        self.VAL_FOLD = self.DATA_ROOT / "fold2"    # As per paper  
        self.TEST_FOLD = self.DATA_ROOT / "fold3"   # As per paper
        
        # Model parameters matching paper
        self.NUM_CLASSES = 5
        self.CLASS_NAMES = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]
        
        # Feature dimensions exactly as in paper
        self.CNN_FEATURE_DIM = 768    # DINOv2 features (Section 3.2.1)
        self.MORPH_FEATURE_DIM = 6    # Morphological features (Section 3.2.2)
        self.NUCLEAR_FEATURE_DIM = 12 # StarDist features (Section 3.2.3)
        self.CELL_FEATURE_DIM = self.CNN_FEATURE_DIM + self.MORPH_FEATURE_DIM + self.NUCLEAR_FEATURE_DIM  # Total 786
        
        # HiGATE architecture parameters (Section 3.4)
        self.HIDDEN_DIM = 128         # Hidden dimension for GNN layers
        self.NUM_LAYERS = 3           # Number of GNN layers
        self.NUM_HEADS = 8            # Number of attention heads
        self.DROPOUT_RATE = 0.2       # Dropout rate
        self.CLUSTERING_DIM = 64      # Feature projection dimension for clustering
        
        # Graph construction parameters (Section 3.3)
        self.K_MIN = 3                # Minimum k for k-NN
        self.K_MAX = 15               # Maximum k for k-NN  
        self.MIN_CLUSTER_SIZE = 3     # Minimum cluster size for DBSCAN
        self.CLUSTERING_EPS = 'adaptive'  # Adaptive epsilon
        
        # Training parameters (Section 4.1)
        self.BATCH_SIZE = 16
        self.LEARNING_RATE = 1e-4
        self.WEIGHT_DECAY = 1e-5
        self.EPOCHS = 100
        self.EARLY_STOPPING_PATIENCE = 20
        self.FOCAL_LOSS_GAMMA = 2.0   # Focal loss parameter
        
        # Output directories
        self.RESULTS_PATH = Path("results")
        self.MODEL_SAVE_PATH = Path("saved_models")
        self.LOG_DIR = Path("logs")
        self.GRAPH_DATA_PATH = Path("graph_data")
        
        # Initialize with default morphological stats
        self.MORPH_MEAN = [0.0] * self.MORPH_FEATURE_DIM
        self.MORPH_STD = [1.0] * self.MORPH_FEATURE_DIM
        
        # Setup directories and validate paths
        self._initialize()

    def _initialize(self):
        """Ensure directories exist and validate paths"""
        # Create output directories
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        self.RESULTS_PATH.mkdir(parents=True, exist_ok=True)
        self.GRAPH_DATA_PATH.mkdir(parents=True, exist_ok=True)
        
        # Validate dataset paths
        for fold in [self.TRAIN_FOLD, self.VAL_FOLD, self.TEST_FOLD]:
            if not fold.exists():
                logger.warning(f"Dataset folder not found: {fold}")
                # Create dummy structure for testing
                try:
                    (fold / "images").mkdir(parents=True, exist_ok=True)
                    (fold / "masks").mkdir(parents=True, exist_ok=True)
                except:
                    pass

    def to_dict(self):
        """Convert config to dictionary for saving"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

# Global config instance
config = Config()
