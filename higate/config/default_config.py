"""Default configuration for HiGATE model and training."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class ModelConfig:
    """HiGATE model configuration."""
    # Model architecture
    num_classes: int = 5
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.2
    
    # Feature dimensions
    visual_dim: int = 256
    morph_dim: int = 128
    nuclear_dim: int = 128
    fused_dim: int = 512
    
    # Graph parameters
    spatial_decay: float = 50.0  # σ_d in μm
    spatial_weight: float = 0.01
    k_neighbors: int = 20
    min_cells_per_region: int = 50
    max_regions: int = 50
    min_regions: int = 5
    
    # Feature extraction
    fine_tune_dinov2: bool = True
    roi_size: int = 112
    input_size: int = 224

@dataclass
class TrainingConfig:
    """Training configuration."""
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 16
    num_epochs: int = 100
    
    # Loss function
    focal_gamma: float = 2.0
    class_weights: List[float] = field(default_factory=lambda: [0.4338, 0.1821, 0.1524, 0.0541, 0.1786])
    tissue_weights: Optional[List[float]] = None
    
    # Scheduler
    scheduler_t0: int = 10
    scheduler_tmult: int = 2
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    save_frequency: int = 5

@dataclass
class DataConfig:
    """Data configuration."""
    # Dataset paths
    pannuke_root: str = "/data/pannuke"
    monuseg_root: str = "/data/monuseg"
    digestpath_root: str = "/data/digestpath"
    tcga_root: str = "/data/tcga-brca"
    tcga_csv: str = "/data/tcga-brca/labels.csv"
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    # Augmentation
    use_augmentation: bool = True
    rotation_degrees: int = 15
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)

@dataclass
class HiGATEConfig:
    """Master configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    seed: int = 42
    device: str = "cuda"
    experiment_name: str = "higate_experiment"
