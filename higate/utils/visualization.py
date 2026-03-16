"""Visualization utilities for figures and plots."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from typing import List, Optional, Dict, Any
from pathlib import Path

class FigureGenerator:
    """Generate all figures from the paper."""
    
    def __init__(self, save_dir: str = "figures"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
    def plot_roc_curves(self, y_true_list: List[np.ndarray], 
                        y_score_list: List[np.ndarray],
                        model_names: List[str],
                        save_name: str = "roc_curves.png"):
        """Plot ROC curves for multiple models (Fig 2a)."""
        plt.figure(figsize=(8, 6))
        
        for y_true, y_score, name in zip(y_true_list, y_score_list, model_names):
            if y_score.ndim == 2:  # Multi-class
                n_classes = y_score.shape[1]
                fpr = dict()
                tpr = dict()
                
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve((y_true == i).astype(int), y_score[:, i])
                
                # Macro-average
                all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
                mean_tpr = np.zeros_like(all_fpr)
                for i in range(n_classes):
                    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
                mean_tpr /= n_classes
                
                roc_auc = auc(all_fpr, mean_tpr)
                plt.plot(all_fpr, mean_tpr, lw=2, 
                        label=f'{name} (AUROC={roc_auc:.3f})')
            else:
                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, 
                        label=f'{name} (AUROC={roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_pr_curves(self, y_true_list: List[np.ndarray],
                       y_score_list: List[np.ndarray],
                       model_names: List[str],
                       save_name: str = "pr_curves.png"):
        """Plot Precision-Recall curves (Fig 2b)."""
        plt.figure(figsize=(8, 6))
        
        for y_true, y_score, name in zip(y_true_list, y_score_list, model_names):
            if y_score.ndim == 2:  # Multi-class
                n_classes = y_score.shape[1]
                precision = dict()
                recall = dict()
                
                for i in range(n_classes):
                    precision[i], recall[i], _ = precision_recall_curve(
                        (y_true == i).astype(int), y_score[:, i]
                    )
                
                # Macro-average
                all_recall = np.unique(np.concatenate([recall[i] for i in range(n_classes)]))
                mean_precision = np.zeros_like(all_recall)
                for i in range(n_classes):
                    mean_precision += np.interp(all_recall, recall[i][::-1], precision[i][::-1])
                mean_precision /= n_classes
                
                pr_auc = auc(all_recall, mean_precision)
                plt.plot(all_recall, mean_precision, lw=2,
                        label=f'{name} (AUPRC={pr_auc:.3f})')
            else:
                precision, recall, _ = precision_recall_curve(y_true, y_score)
                pr_auc = auc(recall, precision)
                plt.plot(recall, precision, lw=2,
                        label=f'{name} (AUPRC={pr_auc:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_dynamics(self, history: Dict[str, List[float]],
                               save_name: str = "training_dynamics.png"):
        """Plot training and validation curves (Fig 3)."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        axes[0].plot(history['train_acc'], label='Train', lw=2)
        axes[0].plot(history['val_acc'], label='Validation', lw=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].set_title('Accuracy Progression')
        axes[0].grid(True, alpha=0.3)
        
        # Loss
        axes[1].plot(history['train_loss'], label='Train', lw=2)
        axes[1].plot(history['val_loss'], label='Validation', lw=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].set_title('Loss Curves')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_per_class_accuracy(self, class_names: List[str],
                                accuracies_higate: List[float],
                                accuracies_baseline: List[float],
                                baseline_name: str = "HACT-Net",
                                save_name: str = "per_class_accuracy.png"):
        """Plot per-class accuracy comparison (Fig 4)."""
        x = np.arange(len(class_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, accuracies_higate, width, 
                       label='HiGATE', color='#2E86AB', alpha=0.8)
        bars2 = ax.bar(x + width/2, accuracies_baseline, width,
                       label=baseline_name, color='#A23B72', alpha=0.8)
        
        ax.set_ylabel('Accuracy')
        ax.set_title('Per-Class Accuracy Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_ablation(self, component_names: List[str],
                      accuracies: List[float],
                      save_name: str = "ablation.png"):
        """Plot ablation study results (Fig 6)."""
        y_pos = np.arange(len(component_names))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(y_pos, accuracies, color='#2E86AB', alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(component_names)
        ax.set_xlabel('Accuracy')
        ax.set_title('Ablation Study')
        ax.set_xlim(0.8, 0.95)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, accuracies)):
            ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', ha='left', fontsize=9)
        
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_computational_efficiency(self, model_names: List[str],
                                      params: List[float],
                                      inference_times: List[float],
                                      save_name: str = "efficiency.png"):
        """Plot computational efficiency comparison (Fig 5)."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Parameters
        bars1 = axes[0].bar(model_names, params, color='#2E86AB', alpha=0.8)
        axes[0].set_ylabel('Parameters (Millions)')
        axes[0].set_title('Model Size')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            axes[0].annotate(f'{height:.1f}M',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', va='bottom', fontsize=8)
        
        # Inference time
        bars2 = axes[1].bar(model_names, inference_times, color='#A23B72', alpha=0.8)
        axes[1].set_ylabel('Inference Time (ms)')
        axes[1].set_title('Inference Speed')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            axes[1].annotate(f'{height:.1f}ms',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_explainability(self, image: np.ndarray,
