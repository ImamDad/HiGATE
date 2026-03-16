"""Training loop and utilities."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
import logging
from typing import Dict, Optional, Any
import wandb

from ..models.higate import HiGATE
from .losses import HiGATELoss
from .metrics import MetricsCalculator

logger = logging.getLogger(__name__)

class Trainer:
    """Trainer for HiGATE model."""
    
    def __init__(self, model: HiGATE, config, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Setup loss
        self.criterion = HiGATELoss(config.training)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.training.scheduler_t0,
            T_mult=config.training.scheduler_tmult
        )
        
        # Metrics calculator
        self.metrics = MetricsCalculator(
            num_classes=config.model.num_classes,
            class_names=config.data.class_names if hasattr(config.data, 'class_names') else None
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_metrics = []
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch} [Train]")
        
        for batch in pbar:
            # Move data to device
            images = batch['images'].to(self.device)
            morph = batch['morph'].to(self.device)
            stardist = batch['stardist'].to(self.device)
            positions = batch['positions'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(images, morph, stardist, positions)
            
            # Compute loss
            losses = self.criterion(
                output['logits'], labels,
                output['S'], positions,
                output.get('edge_index', None)  # Need to pass edge_index
            )
            
            # Backward pass
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update metrics
            total_loss += losses['total'].item()
            
            preds = output['logits'].argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total'].item(),
                'cls_loss': losses['classification'].item(),
                'spatial_loss': losses['spatial'].item()
            })
        
        # Compute epoch metrics
        epoch_loss = total_loss / len(train_loader)
        train_metrics = self.metrics.classification_metrics(
            np.array(all_labels), np.array(all_preds)
        )
        train_metrics['loss'] = epoch_loss
        
        return train_metrics
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch} [Val]")
        
        for batch in pbar:
            # Move data to device
            images = batch['images'].to(self.device)
            morph = batch['morph'].to(self.device)
            stardist = batch['stardist'].to(self.device)
            positions = batch['positions'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            output = self.model(images, morph, stardist, positions)
            
            # Compute loss
            losses = self.criterion(
                output['logits'], labels,
                output['S'], positions,
                output.get('edge_index', None)
            )
            
            total_loss += losses['total'].item()
            
            # Store predictions
            probs = torch.softmax(output['logits'], dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
        
        # Compute validation metrics
        val_loss = total_loss / len(val_loader)
        val_metrics = self.metrics.classification_metrics(
            np.array(all_labels), np.array(all_preds), np.array(all_probs)
        )
        val_metrics['loss'] = val_loss
        
        return val_metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int, use_wandb: bool = False):
        """Main training loop."""
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_metrics)
            
            # Validate
            val_metrics = self.validate(val_loader)
            self.val_metrics.append(val_metrics)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics.get('accuracy', 0):.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics.get('accuracy', 0):.4f}"
            )
            
            if use_wandb:
                wandb.log({
                    'train/loss': train_metrics['loss'],
                    'train/accuracy': train_metrics.get('accuracy', 0),
                    'val/loss': val_metrics['loss'],
                    'val/accuracy': val_metrics.get('accuracy', 0),
                    'val/f1_macro': val_metrics.get('f1_macro', 0),
                    'val/auroc': val_metrics.get('auroc', 0),
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Save checkpoint
            if val_metrics.get('accuracy', 0) > self.best_val_acc:
                self.best_val_acc = val_metrics.get('accuracy', 0)
                self.save_checkpoint('best_model.pth')
            
            if epoch % self.config.training.save_frequency == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
        
        # Save final model
        self.save_checkpoint('final_model.pth')
        
        return {
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics,
            'best_val_acc': self.best_val_acc
        }
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_metrics = checkpoint.get('val_metrics', [])
        
        logger.info(f"Checkpoint loaded from {path}")
