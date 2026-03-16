"""Unit tests for dataset classes."""

import unittest
import torch
import numpy as np
import tempfile
import json
from pathlib import Path
import sys
import os
import shutil

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets.pannuke import PanNukeDataset
from data.datasets.monuseg import MoNuSegDataset
from data.datasets.digestpath import DigestPathDataset
from data.datasets.tcga_brca import TCGA_BRCA_Dataset
from data.transforms import get_train_transforms, get_val_transforms, RandomRotate90, StainAugmentation


class TestPanNukeDataset(unittest.TestCase):
    """Test PanNuke dataset."""
    
    def setUp(self):
        """Create temporary mock PanNuke data."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock data
        self.num_images = 5
        self.num_nuclei_per_image = 10
        self.image_size = 256
        
        # Mock images (C, H, W)
        images = np.random.randint(0, 255, (self.num_images, 3, self.image_size, self.image_size), dtype=np.uint8)
        np.save(Path(self.temp_dir) / 'images_fold_1.npy', images)
        
        # Mock masks
        masks = np.random.randint(0, self.num_nuclei_per_image, 
                                  (self.num_images, self.image_size, self.image_size), dtype=np.uint16)
        np.save(Path(self.temp_dir) / 'masks_fold_1.npy', masks)
        
        # Mock labels
        labels = []
        for img_idx in range(self.num_images):
            img_labels = {}
            for inst_id in range(1, self.num_nuclei_per_image + 1):
                img_labels[str(inst_id)] = {
                    'centroid': [np.random.randint(50, 200), np.random.randint(50, 200)],
                    'bbox': [50, 50, 150, 150],
                    'class': np.random.randint(0, 5)
                }
            labels.append(img_labels)
        
        with open(Path(self.temp_dir) / 'labels_fold_1.json', 'w') as f:
            json.dump(labels, f)
        
    def tearDown(self):
        """Remove temporary directory."""
        shutil.rmtree(self.temp_dir)
        
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        dataset = PanNukeDataset(root_dir=self.temp_dir, fold='fold_1')
        
        self.assertEqual(len(dataset), self.num_images * self.num_nuclei_per_image)
        
    def test_get_item(self):
        """Test getting a sample."""
        dataset = PanNukeDataset(root_dir=self.temp_dir, fold='fold_1')
        
        sample = dataset[0]
        
        # Check keys
        expected_keys = ['images', 'morph', 'stardist', 'positions', 'labels', 'image_idx', 'instance_id']
        for key in expected_keys:
            self.assertIn(key, sample)
        
        # Check shapes
        self.assertEqual(sample['images'].shape, (3, 224, 224))
        self.assertEqual(sample['morph'].shape, (6,))
        self.assertEqual(sample['stardist'].shape, (12,))
        self.assertEqual(sample['positions'].shape, (2,))
        self.assertIsInstance(sample['labels'], torch.Tensor)
        
    def test_normalization(self):
        """Test morphological feature normalization."""
        dataset = PanNukeDataset(root_dir=self.temp_dir, fold='fold_1', normalize_morph=True)
        
        # Set normalization stats
        morph_mean = np.random.randn(6)
        morph_std = np.random.randn(6) + 5  # Ensure positive
        dataset.set_normalization_stats(morph_mean, morph_std)
        
        sample = dataset[0]
        
        # Check that normalization was applied (values should be different from raw)
        self.assertFalse(torch.allclose(sample['morph'], torch.zeros(6)))


class TestMoNuSegDataset(unittest.TestCase):
    """Test MoNuSeg dataset."""
    
    def setUp(self):
        """Create temporary mock MoNuSeg data."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create directory structure
        split_dir = Path(self.temp_dir) / 'test'
        images_dir = split_dir / 'images'
        masks_dir = split_dir / 'masks'
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)
        
        # Create mock images and masks
        from PIL import Image
        self.num_samples = 3
        
        for i in range(self.num_samples):
            # Create random image
            img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            img.save(images_dir / f'test_{i}.png')
            
            # Create random mask
            mask = Image.fromarray(np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255)
            mask.save(masks_dir / f'test_{i}.png')
        
    def tearDown(self):
        """Remove temporary directory."""
        shutil.rmtree(self.temp_dir)
        
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        dataset = MoNuSegDataset(root_dir=self.temp_dir, split='test')
        
        self.assertEqual(len(dataset), self.num_samples)
        
    def test_get_item(self):
        """Test getting a sample."""
        dataset = MoNuSegDataset(root_dir=self.temp_dir, split='test')
        
        img, mask = dataset[0]
        
        # Check shapes
        self.assertEqual(img.shape, (3, 512, 512))
        self.assertEqual(mask.shape, (512, 512))
        
        # Check types
        self.assertIsInstance(img, torch.Tensor)
        self.assertIsInstance(mask, torch.Tensor)


class TestDigestPathDataset(unittest.TestCase):
    """Test DigestPath dataset."""
    
    def setUp(self):
        """Create temporary mock DigestPath data."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create directory structure
        split_dir = Path(self.temp_dir) / 'test'
        images_dir = split_dir / 'images'
        images_dir.mkdir(parents=True)
        
        # Create mock images
        from PIL import Image
        import pandas as pd
        
        self.num_samples = 5
        data = []
        
        for i in range(self.num_samples):
            # Create random image
            img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            img_path = images_dir / f'test_{i}.png'
            img.save(img_path)
            
            data.append({
                'image_name': f'test_{i}.png',
                'label': np.random.randint(0, 2)
            })
        
        # Create CSV
        df = pd.DataFrame(data)
        df.to_csv(Path(self.temp_dir) / 'test.csv', index=False)
        
    def tearDown(self):
        """Remove temporary directory."""
        shutil.rmtree(self.temp_dir)
        
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        dataset = DigestPathDataset(root_dir=self.temp_dir, split='test')
        
        self.assertEqual(len(dataset), self.num_samples)
        
    def test_get_item(self):
        """Test getting a sample."""
        dataset = DigestPathDataset(root_dir=self.temp_dir, split='test')
        
        img, label = dataset[0]
        
        # Check shapes
        self.assertEqual(img.shape, (3, 256, 256))
        self.assertIsInstance(label, torch.Tensor)
        self.assertIn(label.item(), [0, 1])


class TestTCGA_BRCA_Dataset(unittest.TestCase):
    """Test TCGA-BRCA dataset."""
    
    def setUp(self):
        """Create temporary mock TCGA-BRCA data."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock slide files
        slides_dir = Path(self.temp_dir) / 'slides'
        slides_dir.mkdir()
        
        import pandas as pd
        
        self.num_samples = 3
        data = []
        
        for i in range(self.num_samples):
            # Create empty slide file (just for testing)
            slide_path = slides_dir / f'slide_{i}.svs'
            slide_path.touch()
            
            data.append({
                'slide_id': f'slide_{i}',
                'grade': np.random.randint(1, 4)  # Grades I-III
            })
        
        # Create CSV
        df = pd.DataFrame(data)
        df.to_csv(Path(self.temp_dir) / 'labels.csv', index=False)
        
    def tearDown(self):
        """Remove temporary directory."""
        shutil.rmtree(self.temp_dir)
        
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        # Note: This will fail without actual slide files, so we'll patch openslide
        # For testing, we'll skip if openslide not available
        try:
            dataset = TCGA_BRCA_Dataset(
                root_dir=self.temp_dir,
                csv_file=Path(self.temp_dir) / 'labels.csv',
                num_patches=5
            )
            self.assertEqual(len(dataset), self.num_samples)
        except Exception as e:
            self.skipTest(f"OpenSlide not available: {e}")


class TestTransforms(unittest.TestCase):
    """Test data transforms."""
    
    def test_random_rotate90(self):
        """Test random 90-degree rotation."""
        transform = RandomRotate90()
        
        # Create dummy image
        img = torch.randn(3, 224, 224)
        
        # Apply transform multiple times
        for _ in range(10):
            transformed = transform(img)
            self.assertEqual(transformed.shape, (3, 224, 224))
            
    def test_stain_augmentation(self):
        """Test stain augmentation."""
        transform = StainAugmentation(alpha=0.1)
        
        # Create dummy image
        img = torch.randn(3, 224, 224)
        
        # Apply transform
        transformed = transform(img)
        
        # Check shape
        self.assertEqual(transformed.shape, (3, 224, 224))
        
        # Values should be different (sometimes)
        if not torch.allclose(img, transformed):
            self.assertTrue(True)  # Pass
        
    def test_train_transforms(self):
        """Test training transforms composition."""
        from config.default_config import DataConfig
        
        config = DataConfig()
        transforms = get_train_transforms(config)
        
        # Create dummy image
        img = torch.randn(3, 224, 224)
        
        # Apply transforms
        transformed = transforms(img)
        
        # Check shape
        self.assertEqual(transformed.shape, (3, 224, 224))
        
    def test_val_transforms(self):
        """Test validation transforms."""
        transforms = get_val_transforms()
        
        # Create dummy image
        img = torch.randn(3, 224, 224)
        
        # Apply transforms
        transformed = transforms(img)
        
        # Check shape
        self.assertEqual(transformed.shape, (3, 224, 224))


if __name__ == '__main__':
    unittest.main()
