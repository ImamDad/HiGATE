"""MoNuSeg dataset for segmentation evaluation."""

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

class MoNuSegDataset(Dataset):
    """MoNuSeg dataset for nuclei segmentation.
    
    Folder structure:
    root_dir/
    ├── train/
    │   ├── TCGA-*/          # Multiple TCGA slide folders
    │   │   ├── *.png        # Image files
    │   │   └── *_mask.png   # Corresponding mask files
    │   └── ...
    └── test/
        ├── TCGA-*/
        │   ├── *.png
        │   └── *_mask.png
        └── ...
    
    Alternative structure (from Grand Challenge):
    root_dir/
    ├── train/
    │   ├── images/
    │   │   ├── *.png
    │   │   └── ...
    │   └── masks/
    │       ├── *.png
    │       └── ...
    └── test/
        ├── images/
        │   ├── *.png
        │   └── ...
        └── masks/
            ├── *.png
            └── ...
    """
    
    def __init__(self, root_dir: str, split: str = 'test', transform=None,
                 target_size: Tuple[int, int] = (512, 512), structure: str = 'tcga'):
        """
        Args:
            root_dir: Root directory of MoNuSeg dataset
            split: 'train' or 'test'
            transform: Optional transforms
            target_size: Target size for resizing
            structure: 'tcga' for TCGA folder structure, 'grand' for Grand Challenge structure
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_size = target_size
        self.structure = structure
        
        self.image_paths = []
        self.mask_paths = []
        
        if structure == 'tcga':
            # TCGA folder structure
            split_dir = self.root_dir / split
            for case_dir in sorted(split_dir.glob('TCGA-*')):
                # Find all PNG files that are not masks
                images = [f for f in case_dir.glob('*.png') if not f.name.endswith('_mask.png')]
                for img_path in images:
                    mask_path = case_dir / f"{img_path.stem}_mask.png"
                    if mask_path.exists():
                        self.image_paths.append(img_path)
                        self.mask_paths.append(mask_path)
        
        elif structure == 'grand':
            # Grand Challenge structure
            images_dir = self.root_dir / split / 'images'
            masks_dir = self.root_dir / split / 'masks'
            
            if images_dir.exists() and masks_dir.exists():
                self.image_paths = sorted(images_dir.glob('*.png'))
                self.mask_paths = sorted(masks_dir.glob('*.png'))
            else:
                # Alternative: files directly in split directory with naming convention
                split_dir = self.root_dir / split
                all_files = sorted(split_dir.glob('*.png'))
                self.image_paths = [f for f in all_files if 'mask' not in f.name.lower()]
                self.mask_paths = [f for f in all_files if 'mask' in f.name.lower()]
        
        assert len(self.image_paths) == len(self.mask_paths), \
            f"Number of images ({len(self.image_paths)}) and masks ({len(self.mask_paths)}) must match"
        
        print(f"Loaded {len(self.image_paths)} samples from MoNuSeg {split} set")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx: int):
        # Load image
        img = cv2.imread(str(self.image_paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size)
        
        # Load mask (binary segmentation)
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)
        
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        mask = torch.from_numpy(mask).float()
        
        return img, mask
