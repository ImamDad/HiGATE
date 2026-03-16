"""Base dataset class with common functionality."""

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class BaseHistoDataset(Dataset):
    """Base class for histopathology datasets."""
    
    def __init__(self, root_dir: str, transform=None, cache_features: bool = False):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.cache_features = cache_features
        self.feature_cache = {}
        
    def extract_roi(self, image: np.ndarray, centroid: Tuple[int, int], 
                   roi_size: int = 112, target_size: int = 224) -> np.ndarray:
        """Extract ROI around nucleus centroid."""
        cx, cy = centroid
        half = roi_size // 2
        
        # Get ROI boundaries
        x1 = max(0, cx - half)
        x2 = min(image.shape[1], cx + half)
        y1 = max(0, cy - half)
        y2 = min(image.shape[0], cy + half)
        
        # Extract ROI
        roi = image[y1:y2, x1:x2]
        
        # Pad if necessary
        if roi.shape[0] < roi_size or roi.shape[1] < roi_size:
            pad_top = max(0, roi_size - roi.shape[0]) // 2
            pad_bottom = roi_size - roi.shape[0] - pad_top
            pad_left = max(0, roi_size - roi.shape[1]) // 2
            pad_right = roi_size - roi.shape[1] - pad_left
            
            roi = cv2.copyMakeBorder(
                roi, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=0
            )
        
        # Resize to target size
        roi = cv2.resize(roi, (target_size, target_size))
        
        return roi
    
    def compute_morphological_features(self, mask: np.ndarray) -> np.ndarray:
        """Compute 6 morphological features from nuclear mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return np.zeros(6, dtype=np.float32)
        
        cnt = contours[0]
        
        # Basic features
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        # Eccentricity
        if len(cnt) >= 5:
            (_, _), (MA, ma), _ = cv2.fitEllipse(cnt)
            eccentricity = np.sqrt(1 - (ma/MA)**2) if MA > 0 else 0
        else:
            eccentricity = 0
        
        # Solidity
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Extent
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        rect_area = cv2.contourArea(box)
        extent = area / rect_area if rect_area > 0 else 0
        
        # Orientation
        orientation = rect[2]
        
        return np.array([area, perimeter, eccentricity, solidity, extent, orientation], 
                       dtype=np.float32)
    
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        raise NotImplementedError
