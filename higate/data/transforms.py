"""Data augmentation and transformation utilities."""

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random
import numpy as np

class RandomRotate90:
    """Randomly rotate image by 90 degrees."""
    
    def __call__(self, x):
        if random.random() < 0.5:
            k = random.randint(1, 3)
            x = torch.rot90(x, k, dims=[1, 2])
        return x

class RandomElasticDeform:
    """Elastic deformation for histopathology images."""
    
    def __init__(self, alpha=50, sigma=5):
        self.alpha = alpha
        self.sigma = sigma
    
    def __call__(self, img):
        if random.random() < 0.3:
            # Convert to numpy for elastic deformation
            img_np = img.permute(1, 2, 0).numpy()
            
            # Generate random displacement fields
            random_state = np.random.RandomState(None)
            shape = img_np.shape[:2]
            dx = random_state.rand(*shape) * 2 - 1
            dy = random_state.rand(*shape) * 2 - 1
            
            # Smooth displacement fields
            from scipy.ndimage import gaussian_filter
            dx = gaussian_filter(dx, self.sigma, mode='constant') * self.alpha
            dy = gaussian_filter(dy, self.sigma, mode='constant') * self.alpha
            
            # Create mesh grid
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            indices = (y + dy).reshape(-1), (x + dx).reshape(-1)
            
            # Apply deformation
            from scipy.ndimage import map_coordinates
            deformed = np.zeros_like(img_np)
            for c in range(img_np.shape[2]):
                deformed[:, :, c] = map_coordinates(
                    img_np[:, :, c], indices, order=1, mode='reflect'
                ).reshape(shape)
            
            img = torch.from_numpy(deformed).permute(2, 0, 1).float()
        
        return img

class StainAugmentation:
    """Simulate stain variation in histopathology images."""
    
    def __init__(self, alpha=0.1):
        self.alpha = alpha
    
    def __call__(self, img):
        if random.random() < 0.5:
            # Simple brightness/contrast adjustment as proxy for stain variation
            brightness = 1.0 + random.uniform(-self.alpha, self.alpha)
            contrast = 1.0 + random.uniform(-self.alpha, self.alpha)
            img = F.adjust_brightness(img, brightness)
            img = F.adjust_contrast(img, contrast)
        return img

def get_train_transforms(config):
    """Get training transforms."""
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        RandomRotate90(),
        T.ColorJitter(
            brightness=config.brightness_range,
            contrast=config.contrast_range
        ),
        StainAugmentation(alpha=0.1),
        RandomElasticDeform(alpha=50, sigma=5),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms():
    """Get validation transforms."""
    return T.Compose([
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
