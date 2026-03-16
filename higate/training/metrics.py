"""Evaluation metrics for HiGATE."""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix,
    precision_recall_curve, auc
)
from typing import Dict, List, Optional, Tuple
import warnings

class MetricsCalculator:
    """Compute evaluation metrics for classification and segmentation."""
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        
    def classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                               y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute classification metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
        
        # Per-class metrics
        if self.num_classes <= 10:  # Avoid too many classes
            f1_per_class = f1_score(y_true, y_pred, average=None)
            for i, name in enumerate(self.class_names):
                metrics[f'f1_{name}'] = f1_per_class[i]
        
        # ROC-AUC and PR-AUC if probabilities are provided
        if y_prob is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Multi-class ROC-AUC
                if self.num_classes == 2:
                    metrics['auroc'] = roc_auc_score(y_true, y_prob[:, 1])
                    metrics['auprc'] = average_precision_score(y_true, y_prob[:, 1])
                else:
                    metrics['auroc_macro'] = roc_auc_score(
                        y_true, y_prob, multi_class='ovr', average='macro'
                    )
                    metrics['auroc_weighted'] = roc_auc_score(
                        y_true, y_prob, multi_class='ovr', average='weighted'
                    )
                    
                    # Per-class AUPRC
                    precisions = []
                    recalls = []
                    for i in range(self.num_classes):
                        precision, recall, _ = precision_recall_curve(
                            (y_true == i).astype(int), y_prob[:, i]
                        )
                        pr_auc = auc(recall, precision)
                        metrics[f'auprc_{self.class_names[i]}'] = pr_auc
                        precisions.append(precision)
                        recalls.append(recall)
        
        return metrics
    
    def segmentation_metrics(self, pred_mask: np.ndarray, true_mask: np.ndarray,
                            smooth: float = 1e-6) -> Dict[str, float]:
        """Compute segmentation metrics."""
        pred_flat = pred_mask.flatten()
        true_flat = true_mask.flatten()
        
        # Dice coefficient
        intersection = (pred_flat * true_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)
        
        # IoU (Jaccard)
        union = pred_flat.sum() + true_flat.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        
        # Pixel accuracy
        accuracy = (pred_flat == true_flat).sum() / len(pred_flat)
        
        return {
            'dice': dice,
            'iou': iou,
            'accuracy': accuracy
        }
    
    def compute_aji(self, pred_instances: np.ndarray, true_instances: np.ndarray) -> float:
        """Compute Aggregated Jaccard Index for instance segmentation.
        
        This is a simplified version. For full AJI, you'd need instance matching.
        """
        # Simplified: treat as binary for now
        return self.segmentation_metrics(pred_instances > 0, true_instances > 0)['iou']

class ConfusionMatrixVisualizer:
    """Confusion matrix visualization utilities."""
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute confusion matrix."""
        return confusion_matrix(y_true, y_pred)
    
    def plot(self, cm: np.ndarray, save_path: Optional[str] = None, 
             normalize: bool = True):
        """Plot confusion matrix."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, xticklabels=self.class_names,
                   yticklabels=self.class_names, cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
