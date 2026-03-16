"""Dataset-specific configurations."""

from dataclasses import dataclass

@dataclass
class PanNukeConfig:
    """PanNuke dataset specific configuration."""
    num_classes: int = 5
    class_names: list = field(default_factory=lambda: [
        "Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"
    ])
    folds: list = field(default_factory=lambda: ["fold_1", "fold_2", "fold_3"])
    image_size: tuple = (256, 256)
    magnification: int = 40

@dataclass
class MoNuSegConfig:
    """MoNuSeg dataset configuration."""
    num_classes: int = 1  # binary segmentation
    image_size: tuple = (512, 512)
    organs: list = field(default_factory=lambda: [
        "Breast", "Liver", "Kidney", "Prostate", "Bladder", "Colon", "Stomach"
    ])

@dataclass
class DigestPathConfig:
    """DigestPath dataset configuration."""
    num_classes: int = 2  # benign/malignant
    image_size: tuple = (256, 256)
    class_names: list = field(default_factory=lambda: ["Benign", "Malignant"])

@dataclass
class TCGA_BRCA_Config:
    """TCGA-BRCA dataset configuration."""
    num_classes: int = 3  # grade I-III
    class_names: list = field(default_factory=lambda: ["Grade I", "Grade II", "Grade III"])
    num_patches: int = 20
    patch_size: int = 256
    magnification: int = 20
