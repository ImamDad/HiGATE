import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import logging

logger = logging.getLogger(__name__)

class Metrics:
    """Comprehensive metrics calculation matching paper evaluation"""
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.all_predictions = []
        self.all_targets = []
        self.all_probs = []
    
    def update(self, outputs: torch.Tensor, targets: torch.Tensor):
        """
        Update metric accumulators
        
        Args:
            outputs: model predictions (logits) [batch_size, num_classes]
            targets: ground truth labels [batch_size, num_classes] or [batch_size]
        """
        probs = torch.softmax(outputs, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        
        # Handle both one-hot and class index targets
        if targets.dim() > 1 and targets.size(1) > 1:
            targets = torch.argmax(targets, dim=-1)
        
        self.all_predictions.extend(preds.cpu().numpy())
        self.all_targets.extend(targets.cpu().numpy())
        self.all_probs.extend(probs.cpu().numpy())
    
    def compute(self) -> dict:
        """
        Compute all metrics as reported in paper
        
        Returns:
            Dictionary with accuracy, F1-score, AUROC, AUPRC, etc.
        """
        if not self.all_predictions:
            return {
                'accuracy': 0.0,
                'f1_score': 0.0,
                'auc_roc': 0.0,
                'precision': 0.0,
                'recall': 0.0
            }
        
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_predictions)
        y_probs = np.array(self.all_probs)
        
        # Multi-class metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # AUC-ROC (one-vs-rest)
        try:
            if self.num_classes > 1:
                auc_roc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted')
            else:
                auc_roc = roc_auc_score(y_true, y_probs[:, 1])
        except:
            auc_roc = 0.0
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'precision': precision,
            'recall': recall
        }
    
    def compute_detailed_metrics(self) -> dict:
        """Compute per-class metrics for detailed analysis"""
        if not self.all_predictions:
            return {}
        
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_predictions)
        
        per_class_metrics = {}
        for class_idx in range(self.num_classes):
            class_mask = y_true == class_idx
            if np.sum(class_mask) > 0:
                class_accuracy = accuracy_score(y_true[class_mask], y_pred[class_mask])
                class_precision = precision_score(y_true, y_pred, labels=[class_idx], average='micro', zero_division=0)
                class_recall = recall_score(y_true, y_pred, labels=[class_idx], average='micro', zero_division=0)
                class_f1 = f1_score(y_true, y_pred, labels=[class_idx], average='micro', zero_division=0)
                
                per_class_metrics[f'class_{class_idx}'] = {
                    'accuracy': class_accuracy,
                    'precision': class_precision,
                    'recall': class_recall,
                    'f1_score': class_f1,
                    'support': np.sum(class_mask)
                }
        
        return per_class_metrics
