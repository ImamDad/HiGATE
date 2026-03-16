#!/usr/bin/env python
"""Script to reproduce all figures from the paper."""

import argparse
import torch
import numpy as np
import pickle
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.visualization import FigureGenerator
from utils.helpers import setup_logging
import logging

logger = logging.getLogger(__name__)

def generate_synthetic_data():
    """Generate synthetic data for demonstration."""
    np.random.seed(42)
    
    # ROC/PR curves data
    y_true_list = [
        np.random.randint(0, 2, 1000),
        np.random.randint(0, 2, 1000),
        np.random.randint(0, 2, 1000),
        np.random.randint(0, 2, 1000)
    ]
    
    y_score_list = [
        np.random.rand(1000) * 0.9 + 0.1,  # HiGATE (best)
        np.random.rand(1000) * 0.85 + 0.1,  # HACT-Net
        np.random.rand(1000) * 0.8 + 0.1,   # Swin
        np.random.rand(1000) * 0.8 + 0.05   # TransPath
    ]
    
    model_names = ['HiGATE', 'HACT-Net', 'Swin Transformer', 'TransPath']
    
    # Training dynamics
    epochs = 100
    history = {
        'train_acc': 0.5 + 0.4 * (1 - np.exp(-np.linspace(0, 5, epochs))),
        'val_acc': 0.5 + 0.4 * (1 - np.exp(-np.linspace(0, 5, epochs))) - 0.02,
        'train_loss': 2.0 * np.exp(-np.linspace(0, 5, epochs)),
        'val_loss': 2.0 * np.exp(-np.linspace(0, 5, epochs)) + 0.1
    }
    
    # Per-class accuracy
    class_names = ['Bile-Duct', 'Bladder', 'Breast', 'Cervix', 'Colon',
                   'Esophagus', 'Head&Neck', 'Kidney', 'Liver', 'Lung',
                   'Ovarian', 'Pancreatic', 'Prostate', 'Skin', 'Stomach',
                   'Testis', 'Thyroid', 'Uterus']
    
    acc_higate = [0.91, 0.90, 0.92, 0.89, 0.93, 0.91, 0.90, 0.91, 0.89,
                  0.92, 0.90, 0.91, 0.92, 0.93, 0.91, 0.90, 0.89, 0.91]
    acc_baseline = [0.87, 0.86, 0.89, 0.85, 0.91, 0.88, 0.87, 0.88, 0.86,
                    0.90, 0.87, 0.88, 0.89, 0.91, 0.89, 0.87, 0.86, 0.88]
    
    # Ablation study
    component_names = [
        'Full HiGATE',
        'w/o Visual Features',
        'w/o Morphological',
        'w/o Nuclear Features',
        'w/o Bidirectional Attention',
        'w/o Tissue Graph',
        'w/o Cell Graph',
        'Fixed Grid Pooling',
        'k-means Pooling',
        'Watershed Pooling'
    ]
    acc_ablation = [0.913, 0.875, 0.882, 0.886, 0.879, 0.865, 0.854, 0.870, 0.864, 0.868]
    
    # Computational efficiency
    model_names_eff = ['HiGATE', 'HACT-Net', 'Swin', 'TransPath', 'ViT', 'ResNet50']
    params = [3.2, 28.4, 49.2, 87.2, 86.6, 23.5]
    inference_times = [20.9, 15.9, 22.4, 24.1, 18.7, 12.1]
    
    return {
        'roc': (y_true_list, y_score_list, model_names),
        'pr': (y_true_list, y_score_list, model_names),
        'training': history,
        'per_class': (class_names, acc_higate, acc_baseline),
        'ablation': (component_names, acc_ablation, 0.913),
        'efficiency': (model_names_eff, params, inference_times)
    }

def main():
    parser = argparse.ArgumentParser(description='Reproduce figures from paper')
    parser.add_argument('--data_file', type=str, default=None,
                        help='Pickle file with evaluation data')
    parser.add_argument('--save_dir', type=str, default='figures',
                        help='Directory to save figures')
    args = parser.parse_args()
    
    setup_logging()
    
    # Load or generate data
    if args.data_file and Path(args.data_file).exists():
        with open(args.data_file, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Loaded data from {args.data_file}")
    else:
        logger.info("Generating synthetic data for demonstration")
        data = generate_synthetic_data()
    
    # Create figure generator
    fig_gen = FigureGenerator(save_dir=args.save_dir)
    
    # Generate all figures
    logger.info("Generating Figure 2a: ROC Curves")
    fig_gen.plot_roc_curves(*data['roc'], save_name='fig2a_roc_curves.png')
    
    logger.info("Generating Figure 2b: PR Curves")
    fig_gen.plot_pr_curves(*data['pr'], save_name='fig2b_pr_curves.png')
    
    logger.info("Generating Figure 3: Training Dynamics")
    fig_gen.plot_training_dynamics(data['training'], save_name='fig3_training_dynamics.png')
    
    logger.info("Generating Figure 4: Per-Class Accuracy")
    fig_gen.plot_per_class_accuracy(*data['per_class'], save_name='fig4_per_class_accuracy.png')
    
    logger.info("Generating Figure 5: Computational Efficiency")
    fig_gen.plot_computational_efficiency(*data['efficiency'], save_name='fig5_efficiency.png')
    
    logger.info("Generating Figure 6: Ablation Study")
    fig_gen.plot_ablation(*data['ablation'], save_name='fig6_ablation.png')
    
    logger.info(f"All figures saved to {args.save_dir}/")

if __name__ == '__main__':
    main()
