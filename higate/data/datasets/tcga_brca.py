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
    """
    TCGA-BRCA dataset with patch sampling for WSI grading.
    
    Folder structure:
    root_dir/
    ├── slides/
    │   ├── TCGA-XX-XXXX-XXZ.XX.svs  # Slide files
    │   ├── TCGA-YY-YYYY-YYZ.YY.svs
    │   └── ...
    ├── labels.csv                    # Contains: slide_id, grade (1/2/3)
    ├── clinical.csv                   # Clinical data (optional)
    └── tissue_masks/                  # Optional tissue segmentation masks
    
    Grade mapping:
    - Grade I: 1 → 0
    - Grade II: 2 → 1
    - Grade III: 3 → 2
    """
    
    def __init__(self, root_dir: str, csv_file: str, transform=None,
                 num_patches: int = 20, patch_size: int = 256,
                 level: int = 0, tissue_threshold: float = 0.1,
                 use_tissue_masks: bool = False):
        """
        Args:
            root_dir: Root directory containing slides/ folder
            csv_file: Path to CSV file with slide_id and grade columns
            transform: Optional transforms for patches
            num_patches: Number of patches to sample per slide
            patch_size: Size of patches (patch_size x patch_size)
            level: OpenSlide level (0 for highest resolution)
            tissue_threshold: Minimum tissue ratio for valid patch
            use_tissue_masks: Whether to use pre-computed tissue masks
        """
        self.root_dir = Path(root_dir)
        self.slides_dir = self.root_dir / 'slides'
        self.csv_file = Path(csv_file)
        self.transform = transform
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.level = level
        self.tissue_threshold = tissue_threshold
        self.use_tissue_masks = use_tissue_masks
        
        if use_tissue_masks:
            self.masks_dir = self.root_dir / 'tissue_masks'
        
        # Load metadata
        self.df = pd.read_csv(self.csv_file)
        
        # Standardize column names
        if 'slide_id' not in self.df.columns:
            if 'filename' in self.df.columns:
                self.df = self.df.rename(columns={'filename': 'slide_id'})
            elif 'slide' in self.df.columns:
                self.df = self.df.rename(columns={'slide': 'slide_id'})
        
        if 'grade' not in self.df.columns:
            if 'label' in self.df.columns:
                self.df = self.df.rename(columns={'label': 'grade'})
            elif 'class' in self.df.columns:
                self.df = self.df.rename(columns={'class': 'grade'})
        
        # Ensure slide_id has .svs extension
        self.df['slide_id'] = self.df['slide_id'].apply(
            lambda x: x if x.endswith('.svs') else x + '.svs'
        )
        
        print(f"Loaded {len(self.df)} slides from TCGA-BRCA")
        
    def _is_tissue_patch(self, slide: openslide.OpenSlide, x: int, y: int) -> bool:
        """Check if patch contains sufficient tissue."""
        patch = slide.read_region((x, y), self.level, (self.patch_size, self.patch_size))
        patch = np.array(patch.convert('L'))  # Convert to grayscale
        
        # Simple tissue detection: percentage of non-white pixels
        tissue_pixels = (patch < 240).sum()
        tissue_ratio = tissue_pixels / (self.patch_size * self.patch_size)
        
        return tissue_ratio > self.tissue_threshold
    
    def _sample_patches_with_mask(self, slide: openslide.OpenSlide, mask_path: Path) -> List[np.ndarray]:
        """Sample patches using pre-computed tissue mask."""
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        tissue_coords = np.where(mask > 0)
        
        patches = []
        attempts = 0
        max_attempts = self.num_patches * 10
        
        while len(patches) < self.num_patches and attempts < max_attempts:
            # Randomly sample from tissue region
            idx = random.randint(0, len(tissue_coords[0]) - 1)
            y, x = tissue_coords[0][idx], tissue_coords[1][idx]
            
            # Ensure patch fits within slide
            x = min(x, slide.dimensions[0] - self.patch_size)
            y = min(y, slide.dimensions[1] - self.patch_size)
            x = max(0, x)
            y = max(0, y)
            
            patch = slide.read_region((x, y), self.level, (self.patch_size, self.patch_size))
            patch = patch.convert('RGB')
            patch = np.array(patch)
            
            if self.transform:
                patch = self.transform(patch)
            else:
                patch = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
            
            patches.append(patch)
            attempts += 1
        
        return patches
    
    def _sample_patches_random(self, slide: openslide.OpenSlide) -> List[np.ndarray]:
        """Randomly sample patches with tissue detection."""
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
        
        return patches
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        slide_id = row['slide_id']
        grade = row['grade'] - 1  # Convert 1/2/3 to 0/1/2
        
        # Load slide
        slide_path = self.slides_dir / slide_id
        if not slide_path.exists():
            # Try without .svs
            slide_path = self.slides_dir / slide_id.replace('.svs', '')
        
        try:
            slide = openslide.OpenSlide(str(slide_path))
        except Exception as e:
            print(f"Error loading slide {slide_path}: {e}")
            # Return dummy data
            dummy_patches = torch.zeros(self.num_patches, 3, self.patch_size, self.patch_size)
            return dummy_patches, torch.tensor(grade, dtype=torch.long)
        
        # Sample patches
        if self.use_tissue_masks:
            mask_path = self.masks_dir / f"{slide_id.replace('.svs', '')}_mask.png"
            if mask_path.exists():
                patches = self._sample_patches_with_mask(slide, mask_path)
            else:
                patches = self._sample_patches_random(slide)
        else:
            patches = self._sample_patches_random(slide)
        
        # Pad if not enough patches
        while len(patches) < self.num_patches:
            patches.append(torch.zeros(3, self.patch_size, self.patch_size))
        
        patches = torch.stack(patches[:self.num_patches])
        
        return patches, torch.tensor(grade, dtype=torch.long)
