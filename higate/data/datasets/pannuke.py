"""PanNuke dataset implementation."""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import cv2
from .base_dataset import BaseHistoDataset

class PanNukeDataset(BaseHistoDataset):
    """PanNuke dataset for nuclei classification.
    
    Folder structure:
    root_dir/
    ├── fold0/
    │   ├── extracted_images_npy/          # Contains image_*.npy files (256,256,3)
    │   ├── extracted_masks/                # Contains mask_*.npy files (256,256,6)
    │   ├── extracted_cell_counts.csv       # Columns: Image,Neoplastic,Inflammatory,Connective,Dead,Epithelial
    │   └── extracted_types.csv              # Columns: img,type (tissue type codes)
    ├── fold1/
    │   ├── extracted_images_npy/
    │   ├── extracted_masks/
    │   ├── extracted_cell_counts.csv
    │   └── extracted_types.csv
    └── fold2/
        ├── extracted_images_npy/
        ├── extracted_masks/
        ├── extracted_cell_counts.csv
        └── extracted_types.csv
    
    Mask channels (6):
    - Channel 0: Neoplastic
    - Channel 1: Inflammatory
    - Channel 2: Connective
    - Channel 3: Dead
    - Channel 4: Epithelial
    - Channel 5: Background
    """
    
    # Class mapping
    CLASS_NAMES = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]
    CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    
    def __init__(self, root_dir: str, fold: str = 'fold0', transform=None,
                 cache_features: bool = False, normalize_morph: bool = True,
                 tissue_type_filter: Optional[List[int]] = None):
        """
        Args:
            root_dir: Root directory containing fold0/, fold1/, fold2/
            fold: Which fold to use ('fold0', 'fold1', 'fold2')
            transform: Optional transforms
            cache_features: Whether to cache features
            normalize_morph: Whether to normalize morphological features
            tissue_type_filter: Optional list of tissue type codes to filter by
        """
        super().__init__(root_dir, transform, cache_features)
        self.fold = fold
        self.normalize_morph = normalize_morph
        self.tissue_type_filter = tissue_type_filter
        
        # Set paths
        self.fold_dir = Path(root_dir) / fold
        self.images_dir = self.fold_dir / 'extracted_images_npy'
        self.masks_dir = self.fold_dir / 'extracted_masks'
        self.cell_counts_path = self.fold_dir / 'extracted_cell_counts.csv'
        self.types_path = self.fold_dir / 'extracted_types.csv'
        
        # Load metadata
        self.cell_counts = pd.read_csv(self.cell_counts_path)
        self.tissue_types = pd.read_csv(self.types_path)
        
        # Get all image files
        self.image_files = sorted(self.images_dir.glob('*.npy'))
        self.mask_files = sorted(self.masks_dir.glob('*.npy'))
        
        # Verify matching files
        image_names = [f.name for f in self.image_files]
        mask_names = [f.name for f in self.mask_files]
        assert image_names == mask_names, "Image and mask files must match"
        
        # Filter by tissue type if specified
        if tissue_type_filter:
            valid_images = self.tissue_types[self.tissue_types['type'].isin(tissue_type_filter)]['img'].tolist()
            self.image_files = [f for f in self.image_files if f.name in valid_images]
            self.mask_files = [f for f in self.mask_files if f.name in valid_images]
            self.cell_counts = self.cell_counts[self.cell_counts['Image'].isin(valid_images)]
            self.tissue_types = self.tissue_types[self.tissue_types['img'].isin(valid_images)]
        
        print(f"Loaded {len(self.image_files)} images from {fold}")
        
        # Build nucleus index
        self.nuclei_list = self._build_nuclei_list()
        print(f"Total nuclei: {len(self.nuclei_list)}")
        
        # Statistics for normalization
        self.morph_mean = None
        self.morph_std = None
        
    def _build_nuclei_list(self):
        """Build list of all nuclei with their metadata."""
        nuclei_list = []
        
        for img_idx, (img_path, mask_path) in enumerate(zip(self.image_files, self.mask_files)):
            # Load mask to get instance information
            mask = np.load(mask_path)  # Shape: (256, 256, 6)
            
            # Get tissue type for this image
            img_name = img_path.name
            tissue_type = self.tissue_types[self.tissue_types['img'] == img_name]['type'].values[0]
            
            # For each class channel (0-4, excluding background at channel 5)
            for class_idx in range(5):  # 0: Neoplastic, 1: Inflammatory, 2: Connective, 3: Dead, 4: Epithelial
                class_mask = mask[:, :, class_idx]
                
                # Find connected components (instances) in this class
                num_labels, labels = cv2.connectedComponents(class_mask.astype(np.uint8))
                
                # For each instance (skip label 0 which is background)
                for inst_id in range(1, num_labels):
                    inst_mask = (labels == inst_id).astype(np.uint8)
                    
                    # Find centroid
                    y, x = np.where(inst_mask > 0)
                    if len(y) > 0:
                        centroid = (int(np.mean(x)), int(np.mean(y)))
                        
                        nuclei_list.append({
                            'image_idx': img_idx,
                            'image_name': img_name,
                            'instance_id': inst_id,
                            'class_idx': class_idx,
                            'class_name': self.CLASS_NAMES[class_idx],
                            'centroid': centroid,
                            'tissue_type': tissue_type
                        })
        
        return nuclei_list
    
    def set_normalization_stats(self, morph_mean: np.ndarray, morph_std: np.ndarray):
        """Set morphological feature statistics for normalization."""
        self.morph_mean = morph_mean
        self.morph_std = morph_std
    
    def __len__(self):
        return len(self.nuclei_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.nuclei_list[idx]
        
        # Load image and mask
        img_path = self.image_files[item['image_idx']]
        mask_path = self.mask_files[item['image_idx']]
        
        img = np.load(img_path)  # Shape: (256, 256, 3)
        mask = np.load(mask_path)  # Shape: (256, 256, 6)
        
        # Extract ROI around nucleus (112x112 as per paper)
        cx, cy = item['centroid']
        roi = self.extract_roi(img, (cx, cy), roi_size=112, target_size=224)
        
        # Extract instance mask for this nucleus
        class_mask = mask[:, :, item['class_idx']]
        _, labels = cv2.connectedComponents(class_mask.astype(np.uint8))
        inst_mask = (labels == item['instance_id']).astype(np.uint8)
        
        # Compute morphological features (Equation 3)
        morph_features = self.compute_morphological_features(inst_mask)
        
        # Normalize if statistics are available
        if self.normalize_morph and self.morph_mean is not None:
            morph_features = (morph_features - self.morph_mean) / (self.morph_std + 1e-8)
        
        # StarDist features (to be replaced with actual extraction)
        # In practice, you'd run StarDist on the image and extract features per nucleus
        # For now, we'll use random features as placeholder
        stardist_features = np.random.randn(12).astype(np.float32)
        
        # Convert ROI to tensor
        roi = torch.from_numpy(roi).permute(2, 0, 1).float() / 255.0
        
        if self.transform:
            roi = self.transform(roi)
        
        return {
            'images': roi,
            'morph': torch.from_numpy(morph_features).float(),
            'stardist': torch.from_numpy(stardist_features).float(),
            'positions': torch.tensor([cx, cy], dtype=torch.float32),
            'labels': torch.tensor(item['class_idx'], dtype=torch.long),
            'image_idx': item['image_idx'],
            'image_name': item['image_name'],
            'instance_id': item['instance_id'],
            'tissue_type': torch.tensor(item['tissue_type'], dtype=torch.long)
        }
    
    def get_tissue_type_distribution(self) -> Dict[int, int]:
        """Get distribution of tissue types in dataset."""
        tissue_counts = {}
        for item in self.nuclei_list:
            tissue_type = item['tissue_type']
            tissue_counts[tissue_type] = tissue_counts.get(tissue_type, 0) + 1
        return tissue_counts
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of nuclear classes."""
        class_counts = {name: 0 for name in self.CLASS_NAMES}
        for item in self.nuclei_list:
            class_counts[item['class_name']] += 1
        return class_counts
