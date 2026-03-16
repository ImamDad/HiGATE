"""MoNuSeg dataset for segmentation evaluation."""

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

class MoNuSegDataset(Dataset):
    """MoNuSeg dataset for nuclei segmentation."""
    
    def __init__(self, root_dir: str, split: str = 'test', transform=None,
                 target_size: Tuple[int, int] = (512, 512)):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_size = target_size
        
        # Get image and mask paths
        self.image_paths = sorted((self.root_dir / split / 'images').glob('*.png'))
        self.mask_paths = sorted((self.root_dir / split / 'masks').glob('*.png'))
        
        assert len(self.image_paths) == len(self.mask_paths), \
            "Number of images and masks must match"
    
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
