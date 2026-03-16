"""DigestPath dataset for colon polyp classification."""

import torch
from torch.utils.data import Dataset
import cv2
import pandas as pd
from pathlib import Path
from typing import Optional

class DigestPathDataset(Dataset):
    """DigestPath dataset for binary classification (benign/malignant).
    
    Folder structure:
    root_dir/
    ├── train/
    │   ├── images/
    │   │   ├── *.bmp
    │   │   └── ...
    │   └── labels/
    │       ├── *.bmp  # Ground truth masks
    │       └── ...
    ├── test/
    │   ├── images/
    │   │   ├── *.bmp
    │   │   └── ...
    │   └── labels/
    │       ├── *.bmp
    │       └── ...
    ├── train.csv          # Contains: image_name, label (0/1)
    └── test.csv           # Contains: image_name, label (0/1)
    
    Note: DigestPath uses .bmp format for images and masks.
    """
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None,
                 image_size: int = 256):
        """
        Args:
            root_dir: Root directory of DigestPath dataset
            split: 'train' or 'test'
            transform: Optional transforms
            image_size: Size to resize images to
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        
        # Load metadata
        csv_path = self.root_dir / f'{split}.csv'
        if not csv_path.exists():
            # Try alternative naming
            csv_path = self.root_dir / f'{split}_labels.csv'
        
        self.df = pd.read_csv(csv_path)
        
        # Set image directory
        self.img_dir = self.root_dir / split / 'images'
        if not self.img_dir.exists():
            # Try alternative structure
            self.img_dir = self.root_dir / split
        
        print(f"Loaded {len(self.df)} samples from DigestPath {split} set")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        
        # Get image name (handle different column names)
        if 'image_name' in row:
            img_name = row['image_name']
        elif 'filename' in row:
            img_name = row['filename']
        else:
            img_name = row.iloc[0]
        
        # Ensure .bmp extension
        if not img_name.endswith('.bmp'):
            img_name = img_name + '.bmp'
        
        # Load and preprocess image
        img_path = self.img_dir / img_name
        img = cv2.imread(str(img_path))
        
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        
        # Get label (handle different column names)
        if 'label' in row:
            label = row['label']
        elif 'class' in row:
            label = row['class']
        else:
            label = row.iloc[1]  # Assume second column is label
        
        # Convert label to int (0: benign, 1: malignant)
        if isinstance(label, str):
            label = 0 if label.lower() in ['benign', '0'] else 1
        
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        return img, torch.tensor(label, dtype=torch.long)
