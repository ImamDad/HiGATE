
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Tuple, List
import logging
from config import config

logger = logging.getLogger(__name__)

class PanNukeDataset(Dataset):
    """
    PanNuke Dataset loader following paper specifications
    Uses folds 1,2,3 for train/val/test as in paper Section 4.1
    """
    
    def __init__(self, fold_path: Path, transform=None, split: str = 'train'):
        self.fold_path = Path(fold_path)
        self.transform = transform
        self.split = split
        
        # Dataset paths as per paper
        self.images_dir = self.fold_path / "extracted_images_npy"
        self.masks_dir = self.fold_path / "extracted_masks" 
        self.labels_path = self.fold_path / "cell_counts.csv"
        
        self._validate_paths()
        self._load_dataset()
        
        logger.info(f"Loaded {split} dataset from {fold_path} with {len(self)} samples")

    def _validate_paths(self):
        """Verify all required paths exist"""
        required = {
            "Images": self.images_dir,
            "Masks": self.masks_dir,
            "Labels": self.labels_path
        }
        for name, path in required.items():
            if not path.exists():
                raise FileNotFoundError(f"{name} directory not found: {path}")

    def _load_dataset(self):
        """Load and prepare dataset with class balancing"""
        # Load image files
        image_files = {f.stem: f for f in self.images_dir.glob("*.npy")}
        
        # Load labels CSV
        self.labels_df = pd.read_csv(self.labels_path)
        
        # Create image_base column by removing .npy extension
        self.labels_df['image_base'] = self.labels_df['Image'].str.replace(r'\.npy$', '', regex=True)
        
        # Filter to only include images that exist
        self.labels_df = self.labels_df[self.labels_df['image_base'].isin(image_files.keys())]
        
        # Get cell counts as labels (5 classes as in paper)
        self.cell_counts = self.labels_df[['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']].values
        
        # Create paths list
        self.image_paths = [image_files[name] for name in self.labels_df['image_base']]
        self.mask_paths = [self.masks_dir / f"{name}.npy" for name in self.labels_df['image_base']]
        
        # Calculate class weights for balanced loss (Section 4.1)
        self.class_weights = self._calculate_class_weights()

    def _calculate_class_weights(self):
        """Calculate class weights for balanced focal loss"""
        total_counts = self.cell_counts.sum(axis=0)
        class_weights = total_counts.max() / (total_counts + 1e-6)
        return torch.tensor(class_weights, dtype=torch.float32)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        try:
            # Load as numpy arrays first
            image = np.load(self.image_paths[idx])
            mask = np.load(self.mask_paths[idx])
            
            # Process mask according to PanNuke format
            if mask.ndim == 3 and mask.shape[2] == 6:  # PanNuke format
                processed_mask = mask.argmax(axis=2)
            else:
                processed_mask = mask if mask.ndim == 2 else mask[..., 0]
            
            # Convert to proper types
            processed_mask = (processed_mask * 255).astype(np.uint8) if processed_mask.max() <= 1 else processed_mask.astype(np.uint8)
            image = image.astype(np.float32) / 255.0  # Normalize to [0,1]
            
            # Apply transforms if specified
            if self.transform:
                image, processed_mask = self.transform(image, processed_mask)
            
            return {
                "image": image,
                "mask": processed_mask,
                "label": self.cell_counts[idx],
                "image_id": self.labels_df.iloc[idx]['image_base'],
                "class_weights": self.class_weights
            }
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {str(e)}")
            return self._get_empty_sample()

    def _get_empty_sample(self):
        """Return an empty sample for error cases"""
        return {
            "image": np.zeros((256, 256, 3), dtype=np.float32),
            "mask": np.zeros((256, 256), dtype=np.uint8),
            "label": np.zeros(5, dtype=np.float32),
            "image_id": "error_sample",
            "class_weights": torch.ones(5, dtype=torch.float32)
        }

def get_collate_fn():
    """Custom collate function for batching"""
    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        
        # Convert to tensors only at batch level
        images = torch.stack([torch.from_numpy(item['image']).permute(2, 0, 1).float() for item in batch])
        masks = torch.stack([torch.from_numpy(item['mask']).long() for item in batch])
        labels = torch.from_numpy(np.stack([item['label'] for item in batch])).float()
        image_ids = [item['image_id'] for item in batch]
        class_weights = batch[0]['class_weights']  # Same for all samples
        
        return {
            'images': images,
            'masks': masks,
            'labels': labels,
            'image_ids': image_ids,
            'class_weights': class_weights
        }
    return collate_fn
