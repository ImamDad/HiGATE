#!/usr/bin/env python
"""Evaluation script for HiGATE model."""

import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.default_config import HiGATEConfig
from data.datasets.pannuke import PanNukeDataset
from data.transforms import get_val_transforms
from models.higate import HiGATE
from training.metrics import MetricsCalculator
from utils.helpers import setup_logging
import logging

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate HiGATE model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to PanNuke dataset')
    parser.add_argument('--fold', type=str, default='fold_1',
                        help='Fold to evaluate on')
    parser.add_argument('--output_file', type=str, default='results.json',
                        help='Output file for results')
    return parser.parse_args()

def main():
    args = parse_args()
    setup_logging()
    
    # Load config from checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['config']
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset
    transforms = get_val_transforms()
    dataset = PanNukeDataset(
        root_dir=args.data_dir,
        fold=args.fold,
        transform=transforms
    )
    
    # Load normalization stats if available
    if hasattr(checkpoint, 'morph_mean'):
        dataset.set_normalization_stats(
            checkpoint['morph_mean'],
            checkpoint['morph_std']
        )
    
    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Create model and load weights
    model = HiGATE(config.model).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Metrics calculator
    metrics_calc = MetricsCalculator(
        num_classes=config.model.num_classes,
        class_names=['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']
    )
    
    # Evaluation
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            images = batch['images'].to(device)
            morph = batch['morph'].to(device)
            stardist = batch['stardist'].to(device)
            positions = batch['positions'].to(device)
            labels = batch['labels'].to(device)
            
            output = model(images, morph, stardist, positions)
            
            probs = torch.softmax(output['logits'], dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
    
    # Compute metrics
    metrics = metrics_calc.classification_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )
    
    # Print results
    logger.info("Evaluation Results:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Results saved to {args.output_file}")

if __name__ == '__main__':
    main()
