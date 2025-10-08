import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class HiGATETrainer:
    """Training class for HiGATE matching paper experimental setup"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.DEVICE
        
        # Class-balanced focal loss (Eq. 1 in paper)
        self.criterion = ClassBalancedFocalLoss(
            num_classes=config.NUM_CLASSES,
            alpha='balanced',
            gamma=config.FOCAL_LOSS_GAMMA
        )
        
        # Optimizer as in paper (Section 4.1)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train(self):
        """Main training loop following paper methodology"""
        from data_processing.dataset import PanNukeDataset
        from training.metrics import Metrics
        
        # Load datasets
        train_dataset = PanNukeDataset(self.config.TRAIN_FOLD)
        val_dataset = PanNukeDataset(self.config.VAL_FOLD)
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True,
            num_workers=self.config.NUM_WORKERS, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False,
            num_workers=self.config.NUM_WORKERS, pin_memory=True
        )
        
        # Training loop
        for epoch in range(self.config.EPOCHS):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader, epoch)
            
            # Log results
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Early stopping
            if self._check_early_stopping(val_metrics['loss']):
                logger.info(f"Early stopping at epoch {epoch}")
                break
                
            self.scheduler.step()
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        metrics = Metrics(self.config.NUM_CLASSES)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} Training")
        for batch_idx, batch in enumerate(pbar):
            # Move data to device and prepare for HiGATE
            batch = self._prepare_batch(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(batch)
            
            # Compute loss
            loss = self.criterion(logits, batch['labels'])
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            metrics.update(logits, batch['labels'])
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        epoch_metrics = metrics.compute()
        epoch_metrics['loss'] = total_loss / len(dataloader)
        
        return epoch_metrics
    
    def validate(self, dataloader, epoch):
        """Validation step"""
        self.model.eval()
        total_loss = 0
        metrics = Metrics(self.config.NUM_CLASSES)
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f"Epoch {epoch} Validation")
            for batch in pbar:
                batch = self._prepare_batch(batch)
                logits = self.model(batch)
                loss = self.criterion(logits, batch['labels'])
                
                total_loss += loss.item()
                metrics.update(logits, batch['labels'])
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        epoch_metrics = metrics.compute()
        epoch_metrics['loss'] = total_loss / len(dataloader)
        
        return epoch_metrics
    
    def _prepare_batch(self, batch):
        """Prepare batch for HiGATE model"""
        # This would convert the batch to the format expected by HiGATE
        # Implementation depends on your dataset structure
        return batch
    
    def _check_early_stopping(self, val_loss):
        """Check early stopping condition"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            self._save_checkpoint('best')
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE
    
    def _save_checkpoint(self, name):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'val_loss': self.best_val_loss
        }
        torch.save(checkpoint, self.config.MODEL_SAVE_PATH / f'{name}_model.pth')
    
    def _log_metrics(self, epoch, train_metrics, val_metrics):
        """Log training and validation metrics"""
        logger.info(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )


class ClassBalancedFocalLoss(nn.Module):
    """Class-balanced focal loss as in paper Eq. 1"""
    
    def __init__(self, num_classes, alpha='balanced', gamma=2.0):
        super(ClassBalancedFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        
        if alpha == 'balanced':
            self.alpha = torch.ones(num_classes) / num_classes
        else:
            self.alpha = alpha
    
    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha[targets] * (1-pt)**self.gamma * BCE_loss
        return focal_loss.mean()
