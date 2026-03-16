"""PanNuke dataset implementation."""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import cv2
from .base_dataset import BaseHistoDataset

class PanNukeDataset(BaseHistoDataset):
    """PanNuke dataset for nuclei classification.
    
    Folder structure:
    root_dir/
    ├── fold0/
    │   ├── extracted_images_npy/          # Contains .npy files for images
    │   ├── extracted_masks/                # Contains .npy files for masks
    │   ├── extracted_cell_counts.csv       # Cell counts per image
    │   └── extracted_types.csv              # Tissue types per image
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
    """
    
    def __init__(self, root_dir: str, fold: str = 'fold0', transform=None,
                 cache_features: bool = False, normalize_morph: bool = True):
        super().__init__(root_dir, transform, cache_features)
        self.fold = fold
        self.normalize_morph = normalize_morph
        
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
        
        assert len(self.image_files) == len(self.mask_files), \
            "Number of images and masks must match"
        
        # Build nucleus index
        self.nuclei_list = self._build_nuclei_list()
        
        # Statistics for normalization
        self.morph_mean = None
        self.morph_std = None
        
    def _build_nuclei_list(self):
        """Build list of all nuclei with their metadata."""
        nuclei_list = []
        
        for img_idx, (img_path, mask_path) in enumerate(zip(self.image_files, self.mask_files)):
            # Load mask to get instance information
            mask = np.load(mask_path)
            unique_instances = np.unique(mask)
            unique_instances = unique_instances[unique_instances > 0]  # Remove background
            
            for inst_id in unique_instances:
                # Get instance mask
                inst_mask = (mask == inst_id).astype(np.uint8)
                
                # Find centroid
                y, x = np.where(inst_mask > 0)
                if len(y) > 0:
                    centroid = (int(np.mean(x)), int(np.mean(y)))
                    
                    # Get class from cell counts (you may need to adjust based on actual label format)
                    # This assumes cell_counts.csv has columns: image_name, instance_id, class
                    row = self.cell_counts[
                        (self.cell_counts['image_name'] == img_path.stem) & 
                        (self.cell_counts['instance_id'] == inst_id)
                    ]
                    
                    if len(row) > 0:
                        cell_class = row.iloc[0]['class']
                    else:
                        # Fallback: random class (replace with actual logic)
                        cell_class = np.random.randint(0, 5)
                    
                    nuclei_list.append({
                        'image_idx': img_idx,
                        'image_name': img_path.stem,
                        'instance_id': int(inst_id),
                        'centroid': centroid,
                        'class': cell_class,
                        'tissue_type': self.tissue_types.iloc[img_idx]['tissue_type']
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
        
        img = np.load(img_path)  # Shape: (H, W, C) or (C, H, W)
        mask = np.load(mask_path)
        
        # Ensure image is in HWC format
        if img.shape[0] == 3 and len(img.shape) == 3:  # CHW format
            img = img.transpose(1, 2, 0)
        
        # Extract ROI around nucleus
        roi = self.extract_roi(img, item['centroid'])
        
        # Extract instance mask
        inst_mask = (mask == item['instance_id']).astype(np.uint8)
        
        # Compute morphological features
        morph_features = self.compute_morphological_features(inst_mask)
        
        # Normalize if statistics are available
        if self.normalize_morph and self.morph_mean is not None:
            morph_features = (morph_features - self.morph_mean) / (self.morph_std + 1e-8)
        
        # StarDist features (to be replaced with actual extraction)
        # In practice, you'd run StarDist on the image and extract features per nucleus
        stardist_features = np.random.randn(12).astype(np.float32)
        
        # Convert to tensor
        roi = torch.from_numpy(roi).permute(2, 0, 1).float() / 255.0
        
        if self.transform:
            roi = self.transform(roi)
        
        return {
            'images': roi,
            'morph': torch.from_numpy(morph_features).float(),
            'stardist': torch.from_numpy(stardist_features).float(),
            'positions': torch.tensor(item['centroid'], dtype=torch.float32),
            'labels': torch.tensor(item['class'], dtype=torch.long),
            'image_idx': item['image_idx'],
            'instance_id': item['instance_id'],
            'tissue_type': item['tissue_type']
        }
