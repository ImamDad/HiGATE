
"""DigestPath dataset for colon polyp classification."""

import torch
from torch.utils.data import Dataset
import cv2
import pandas as pd
from pathlib import Path
from typing import Optional

class DigestPathDataset(Dataset):
    """DigestPath dataset for binary classification."""
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None,
                 image_size: int = 256):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        
        # Load metadata
        csv_path = self.root_dir / f'{split}.csv'
        self.df = pd.read_csv(csv_path)
        self.img_dir = self.root_dir / split / 'images'
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        
        # Load and preprocess image
        img_path = self.img_dir / row['image_name']
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        
        # Get label
        label = row['label']  # 0: benign, 1: malignant
        
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        return img, torch.tensor(label, dtype=torch.long)
