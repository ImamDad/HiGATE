#!/usr/bin/env python
"""Training script for HiGATE model."""

import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
from pathlib import Path
import yaml
import logging
import wandb
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.default_config import HiGATEConfig
from data.datasets.pannuke import PanNukeDataset
from data.transforms import get_train_transforms, get_val_transforms
from models.higate import HiGATE
from training.trainer import Trainer
from utils.helpers import set_seed, setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description='Train HiGATE model')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Configuration file')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to PanNuke dataset (e.g., D:/PanNuke)')
    parser.add_argument('--fold', type=str, default='fold0',
                        choices=['fold0', 'fold1', 'fold2'],
                        help='Cross-validation fold')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint to resume from')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='higate',
                        help='W&B project name')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = HiGATEConfig()
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
            # Update config (simplified - in practice you'd need proper merging)
            for key, value in yaml_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Update paths
    config.data.pannuke_root = args.data_dir
    config.training.checkpoint_dir = args.checkpoint_dir
    
    # Initialize W&B
    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=config, name=f"higate_{args.fold}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets
    train_transforms = get_train_transforms(config.data)
    val_transforms = get_val_transforms()
    
    logger.info(f"Loading PanNuke dataset from {args.data_dir}, fold {args.fold}")
    train_dataset = PanNukeDataset(
        root_dir=args.data_dir,
        fold=args.fold,
        transform=train_transforms
    )
    
    # For validation, use the same fold (PanNuke has predefined splits)
    val_dataset = PanNukeDataset(
        root_dir=args.data_dir,
        fold=args.fold,  # Same fold for validation (adjust as needed)
        transform=val_transforms
    )
    
    # Compute morphological feature statistics for normalization
    logger.info("Computing morphological feature statistics...")
    morph_features = []
    for i in range(min(1000, len(train_dataset))):
        sample = train_dataset[i]
        morph_features.append(sample['morph'].numpy())
    
    morph_mean = np.mean(morph_features, axis=0)
    morph_std = np.std(morph_features, axis=0)
    train_dataset.set_normalization_stats(morph_mean, morph_std)
    val_dataset.set_normalization_stats(morph_mean, morph_std)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        prefetch_factor=config.data.prefetch_factor if hasattr(config.data, 'prefetch_factor') else 2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)} nuclei")
    logger.info(f"Val dataset size: {len(val_dataset)} nuclei")
    
    # Create model
    model = HiGATE(config.model)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = Trainer(model, config, device)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed from checkpoint: {args.resume}")
    
    # Train
    logger.info("Starting training...")
    results = trainer.train(
        train_loader,
        val_loader,
        num_epochs=config.training.num_epochs,
        use_wandb=args.use_wandb
    )
    
    logger.info(f"Training completed. Best validation accuracy: {results['best_val_acc']:.4f}")
    
    if args.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()
