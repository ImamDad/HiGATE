"""Dataset-specific configurations."""

from dataclasses import dataclass, field

@dataclass
class PanNukeConfig:
    """PanNuke dataset specific configuration."""
    num_classes: int = 5
    class_names: list = field(default_factory=lambda: [
        "Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"
    ])
    folds: list = field(default_factory=lambda: ["fold0", "fold1", "fold2"])
    images_subdir: str = "extracted_images_npy"
    masks_subdir: str = "extracted_masks"
    cell_counts_file: str = "extracted_cell_counts.csv"
    types_file: str = "extracted_types.csv"

@dataclass
class MoNuSegConfig:
    """MoNuSeg dataset configuration."""
    num_classes: int = 1  # binary segmentation
    image_size: tuple = (512, 512)
    organs: list = field(default_factory=lambda: [
        "Breast", "Liver", "Kidney", "Prostate", "Bladder", "Colon", "Stomach"
    ])
    structure: str = "tcga"  # 'tcga' or 'grand'

@dataclass
class DigestPathConfig:
    """DigestPath dataset configuration."""
    num_classes: int = 2  # benign/malignant
    image_size: int = 256
    class_names: list = field(default_factory=lambda: ["Benign", "Malignant"])
    image_extension: str = ".bmp"
    train_csv: str = "train.csv"
    test_csv: str = "test.csv"

@dataclass
class TCGA_BRCA_Config:
    """TCGA-BRCA dataset configuration."""
    num_classes: int = 3  # grade I-III
    class_names: list = field(default_factory=lambda: ["Grade I", "Grade II", "Grade III"])
    num_patches: int = 20
    patch_size: int = 256
    level: int = 0
    tissue_threshold: float = 0.1
    slides_subdir: str = "slides"
    labels_file: str = "labels.csv"
