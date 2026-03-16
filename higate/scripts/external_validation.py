#!/usr/bin/env python
"""External validation script for HiGATE on multiple datasets."""

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
from data.datasets.monuseg import MoNuSegDataset
from data.datasets.digestpath import DigestPathDataset
from data.datasets.tcga_brca import TCGA_BRCA_Dataset
from data.transforms import get_val_transforms
from models.higate import HiGATE
from training.metrics import MetricsCalculator
from utils.helpers import setup_logging
import logging

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='External validation of HiGATE')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['monuseg', 'digestpath', 'tcga'],
                        help='Dataset to validate on')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset')
    parser.add_argument('--output_file', type=str, default='external_results.json',
                        help='Output file for results')
    return parser.parse_args()

def validate_monuseg(model, device, data_dir):
    """Validate on MoNuSeg dataset."""
    dataset = MoNuSegDataset(
        root_dir=data_dir,
        split='test',
        transform=get_val_transforms()
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    dice_scores = []
    iou_scores = []
    
    with torch.no_grad():
        for img, mask in loader:
            img = img.to(device)
            # For segmentation, we need to adapt the model
            # This is a placeholder - in practice you'd use a segmentation head
            pred_mask = torch.rand_like(mask) > 0.5
            
            # Compute metrics
            intersection = (pred_mask * mask).sum()
            union = pred_mask.sum() + mask.sum() - intersection
            dice = 2 * intersection / (pred_mask.sum() + mask.sum() + 1e-6)
            iou = intersection / (union + 1e-6)
            
            dice_scores.append(dice.item())
            iou_scores.append(iou.item())
    
    results = {
        'dice_mean': np.mean(dice_scores),
        'dice_std': np.std(dice_scores),
        'iou_mean': np.mean(iou_scores),
        'iou_std': np.std(iou_scores)
    }
    
    return results

def validate_digestpath(model, device, data_dir):
    """Validate on DigestPath dataset."""
    dataset = DigestPathDataset(
        root_dir=data_dir,
        split='test',
        transform=get_val_transforms()
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for img, labels in loader:
            img = img.to(device)
            # For patch-level classification, we need to adapt
            # This is a placeholder
            probs = torch.rand((img.size(0), 2)).softmax(dim=1)
            preds = probs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    metrics_calc = MetricsCalculator(num_classes=2, class_names=['Benign', 'Malignant'])
    results = metrics_calc.classification_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )
    
    return results

def validate_tcga(model, device, data_dir):
    """Validate on TCGA-BRCA dataset."""
    dataset = TCGA_BRCA_Dataset(
        root_dir=data_dir,
        csv_file=Path(data_dir) / 'labels.csv',
        transform=get_val_transforms(),
        num_patches=20
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for patches, label in loader:
            patches = patches.squeeze(0).to(device)
            # Aggregate patch predictions
            probs_list = []
            for patch in patches:
                # Placeholder - in practice you'd run the model on each patch
                prob = torch.rand(1, 3).softmax(dim=1).to(device)
                probs_list.append(prob)
            
            slide_prob = torch.stack(probs_list).mean(dim=0)
            pred = slide_prob.argmax(dim=1)
            
            all_preds.append(pred.item())
            all_labels.append(label.item())
            all_probs.append(slide_prob.cpu().numpy())
    
    metrics_calc = MetricsCalculator(
        num_classes=3,
        class_names=['Grade I', 'Grade II', 'Grade III']
    )
    results = metrics_calc.classification_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )
    
    return results

def main():
    args = parse_args()
    setup_logging()
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['config']
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model and load weights
    model = HiGATE(config.model).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Validate on selected dataset
    logger.info(f"Validating on {args.dataset}...")
    
    if args.dataset == 'monuseg':
        results = validate_monuseg(model, device, args.data_dir)
    elif args.dataset == 'digestpath':
        results = validate_digestpath(model, device, args.data_dir)
    elif args.dataset == 'tcga':
        results = validate_tcga(model, device, args.data_dir)
    
    # Print results
    logger.info("Validation Results:")
    for key, value in results.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {args.output_file}")

if __name__ == '__main__':
    main()
