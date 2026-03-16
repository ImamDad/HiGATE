"""PanNuke dataset implementation."""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Optional, Dict, Any
import cv2
from .base_dataset import BaseHistoDataset

class PanNukeDataset(BaseHistoDataset):
    """PanNuke dataset for nuclei classification."""
    
    def __init__(self, root_dir: str, fold: str = 'fold_1', transform=None,
                 cache_features: bool = False, normalize_morph: bool = True):
        super().__init__(root_dir, transform, cache_features)
        self.fold = fold
        self.normalize_morph = normalize_morph
        
        # Load data
        self.images = np.load(self.root_dir / f'images_{fold}.npy', allow_pickle=True)
        self.masks = np.load(self.root_dir / f'masks_{fold}.npy', allow_pickle=True)
        
        with open(self.root_dir / f'labels_{fold}.json', 'r') as f:
            self.labels = json.load(f)
        
        # Build nucleus index
        self.nuclei_list = self._build_nuclei_list()
        
        # Statistics for normalization
        self.morph_mean = None
        self.morph_std = None
        
    def _build_nuclei_list(self):
        """Build list of all nuclei with their metadata."""
        nuclei_list = []
        for img_idx, label_dict in enumerate(self.labels):
            for inst_id, inst_info in label_dict.items():
                nuclei_list.append({
                    'image_idx': img_idx,
                    'instance_id': int(inst_id),
                    'centroid': tuple(map(int, inst_info['centroid'])),
                    'class': inst_info['class'],
                    'bbox': inst_info.get('bbox', None)
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
        
        # Get image and mask
        img = self.images[item['image_idx']].transpose(1, 2, 0)  # H, W, C
        mask = self.masks[item['image_idx']]
        
        # Extract ROI
        roi = self.extract_roi(img, item['centroid'])
        
        # Extract instance mask
        inst_mask = (mask == item['instance_id']).astype(np.uint8)
        
        # Compute morphological features
        morph_features = self.compute_morphological_features(inst_mask)
        
        # Normalize if statistics are available
        if self.normalize_morph and self.morph_mean is not None:
            morph_features = (morph_features - self.morph_mean) / (self.morph_std + 1e-8)
        
        # StarDist features (to be replaced with actual extraction)
        # In practice, you'd run StarDist on the image and extract features
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
            'instance_id': item['instance_id']
        }
