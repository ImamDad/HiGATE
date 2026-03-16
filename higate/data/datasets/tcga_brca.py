"""TCGA-BRCA dataset for WSI-level breast cancer grading."""

import torch
from torch.utils.data import Dataset
import openslide
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple
import random

class TCGA_BRCA_Dataset(Dataset):
    """TCGA-BRCA dataset with patch sampling for WSI grading."""
    
    def __init__(self, root_dir: str, csv_file: str, transform=None,
                 num_patches: int = 20, patch_size: int = 256,
                 level: int = 0, tissue_threshold: float = 0.1):
        self.root_dir = Path(root_dir)
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.level = level
        self.tissue_threshold = tissue_threshold
        
    def _is_tissue_patch(self, slide: openslide.OpenSlide, x: int, y: int) -> bool:
        """Check if patch contains sufficient tissue."""
        patch = slide.read_region((x, y), self.level, (self.patch_size, self.patch_size))
        patch = np.array(patch.convert('L'))  # Convert to grayscale
        
        # Simple tissue detection: percentage of non-white pixels
        tissue_pixels = (patch < 240).sum()
        tissue_ratio = tissue_pixels / (self.patch_size * self.patch_size)
        
        return tissue_ratio > self.tissue_threshold
    
    def _sample_patches(self, slide: openslide.OpenSlide) -> List[np.ndarray]:
        """Sample patches from tissue regions."""
        width, height = slide.dimensions
        patches = []
        
        attempts = 0
        max_attempts = self.num_patches * 10
        
        while len(patches) < self.num_patches and attempts < max_attempts:
            x = random.randint(0, width - self.patch_size)
            y = random.randint(0, height - self.patch_size)
            
            if self._is_tissue_patch(slide, x, y):
                patch = slide.read_region((x, y), self.level, (self.patch_size, self.patch_size))
                patch = patch.convert('RGB')
                patch = np.array(patch)
                
                if self.transform:
                    patch = self.transform(patch)
                else:
                    patch = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
                
                patches.append(patch)
            
            attempts += 1
        
        # Pad with zeros if not enough patches
        while len(patches) < self.num_patches:
            patches.append(torch.zeros(3, self.patch_size, self.patch_size))
        
        return patches
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        
        # Load slide
        slide_path = self.root_dir / f"{row['slide_id']}.svs"
        slide = openslide.OpenSlide(str(slide_path))
        
        # Sample patches
        patches = self._sample_patches(slide)
        patches = torch.stack(patches)
        
        # Get grade (convert to 0,1,2)
        label = row['grade'] - 1
        
        return patches, torch.tensor(label, dtype=torch.long)
