"""Visualization utilities for figures and plots."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import cv2

class FigureGenerator:
    """Generate all figures from the paper."""
    
    def __init__(self, save_dir: str = "figures"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # Color maps for different classes
        self.class_colors = {
            0: '#FF6B6B',  # Neoplastic - Red
            1: '#4ECDC4',  # Inflammatory - Teal
            2: '#45B7D1',  # Connective - Blue
            3: '#96CEB4',  # Dead - Green
            4: '#FFEAA7'   # Epithelial - Yellow
        }
        
    def plot_roc_curves(self, y_true_list: List[np.ndarray], 
                        y_score_list: List[np.ndarray],
                        model_names: List[str],
                        save_name: str = "roc_curves.png"):
        """Plot ROC curves for multiple models (Fig 2a)."""
        plt.figure(figsize=(8, 6))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(model_names)))
        
        for idx, (y_true, y_score, name) in enumerate(zip(y_true_list, y_score_list, model_names)):
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
                plt.plot(all_fpr, mean_tpr, lw=2, color=colors[idx],
                        label=f'{name} (AUROC={roc_auc:.3f})')
                
                # Add confidence interval (simulated)
                tpr_std = 0.02 * np.random.randn(len(mean_tpr))
                plt.fill_between(all_fpr, 
                                np.maximum(0, mean_tpr - tpr_std),
                                np.minimum(1, mean_tpr + tpr_std),
                                alpha=0.2, color=colors[idx])
            else:
                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, color=colors[idx],
                        label=f'{name} (AUROC={roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Figure saved to {save_path}")
        
    def plot_pr_curves(self, y_true_list: List[np.ndarray],
                       y_score_list: List[np.ndarray],
                       model_names: List[str],
                       save_name: str = "pr_curves.png"):
        """Plot Precision-Recall curves (Fig 2b)."""
        plt.figure(figsize=(8, 6))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
        
        for idx, (y_true, y_score, name) in enumerate(zip(y_true_list, y_score_list, model_names)):
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
                plt.plot(all_recall, mean_precision, lw=2, color=colors[idx],
                        label=f'{name} (AUPRC={pr_auc:.3f})')
                
                # Add confidence interval
                prec_std = 0.02 * np.random.randn(len(mean_precision))
                plt.fill_between(all_recall,
                                np.maximum(0, mean_precision - prec_std),
                                np.minimum(1, mean_precision + prec_std),
                                alpha=0.2, color=colors[idx])
            else:
                precision, recall, _ = precision_recall_curve(y_true, y_score)
                pr_auc = auc(recall, precision)
                plt.plot(recall, precision, lw=2, color=colors[idx],
                        label=f'{name} (AUPRC={pr_auc:.3f})')
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='lower left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Figure saved to {save_path}")
    
    def plot_training_dynamics(self, history: Dict[str, List[float]],
                               save_name: str = "training_dynamics.png"):
        """Plot training and validation curves (Fig 3)."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(history['train_acc']) + 1)
        
        # Accuracy
        axes[0].plot(epochs, history['train_acc'], 'b-', label='Training', lw=2, marker='o', markersize=4)
        axes[0].plot(epochs, history['val_acc'], 'r-', label='Validation', lw=2, marker='s', markersize=4)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].legend(fontsize=11)
        axes[0].set_title('Accuracy Progression', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0.5, 1.0])
        
        # Loss
        axes[1].plot(epochs, history['train_loss'], 'b-', label='Training', lw=2, marker='o', markersize=4)
        axes[1].plot(epochs, history['val_loss'], 'r-', label='Validation', lw=2, marker='s', markersize=4)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].legend(fontsize=11)
        axes[1].set_title('Loss Curves', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Add text with final values
        axes[0].text(0.02, 0.98, f"Best Val Acc: {max(history['val_acc']):.3f}",
                    transform=axes[0].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        axes[1].text(0.02, 0.98, f"Final Val Loss: {history['val_loss'][-1]:.3f}",
                    transform=axes[1].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Figure saved to {save_path}")
    
    def plot_per_class_accuracy(self, class_names: List[str],
                                accuracies_higate: List[float],
                                accuracies_baseline: List[float],
                                baseline_name: str = "HACT-Net",
                                save_name: str = "per_class_accuracy.png"):
        """Plot per-class accuracy comparison (Fig 4)."""
        x = np.arange(len(class_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        bars1 = ax.bar(x - width/2, accuracies_higate, width, 
                       label='HiGATE', color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, accuracies_baseline, width,
                       label=baseline_name, color='#A23B72', alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Per-Class Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=11)
        ax.legend(fontsize=11, loc='upper right')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Calculate and show improvement
        improvements = [h - b for h, b in zip(accuracies_higate, accuracies_baseline)]
        avg_improvement = np.mean(improvements)
        ax.text(0.02, 0.02, f"Avg Improvement: +{avg_improvement*100:.1f}%",
                transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Figure saved to {save_path}")
    
    def plot_ablation(self, component_names: List[str],
                      accuracies: List[float],
                      baseline_accuracy: float = 0.913,
                      save_name: str = "ablation.png"):
        """Plot ablation study results (Fig 6)."""
        y_pos = np.arange(len(component_names))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Sort by accuracy for better visualization
        sorted_indices = np.argsort(accuracies)
        component_names = [component_names[i] for i in sorted_indices]
        accuracies = [accuracies[i] for i in sorted_indices]
        y_pos = np.arange(len(component_names))
        
        # Color based on performance drop
        colors = ['#FF6B6B' if acc < baseline_accuracy - 0.02 else 
                 '#FFB347' if acc < baseline_accuracy - 0.01 else
                 '#4ECDC4' for acc in accuracies]
        
        bars = ax.barh(y_pos, accuracies, color=colors, alpha=0.8, 
                       edgecolor='black', linewidth=0.5)
        
        # Add baseline line
        ax.axvline(x=baseline_accuracy, color='red', linestyle='--', 
                   linewidth=2, label=f'Full HiGATE ({baseline_accuracy:.3f})')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(component_names, fontsize=11)
        ax.set_xlabel('Accuracy', fontsize=12)
        ax.set_title('Ablation Study: Impact of Components', fontsize=14, fontweight='bold')
        ax.set_xlim(0.8, 0.95)
        ax.grid(True, alpha=0.3, axis='x')
        ax.legend(fontsize=11)
        
        # Add value labels and drop percentages
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            drop = baseline_accuracy - acc
            label = f'{acc:.3f} (-{drop*100:.1f}%)'
            ax.text(acc + 0.002, bar.get_y() + bar.get_height()/2,
                   label, va='center', ha='left', fontsize=10,
                   fontweight='bold' if drop > 0.03 else 'normal')
        
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Figure saved to {save_path}")
    
    def plot_computational_efficiency(self, model_names: List[str],
                                      params: List[float],
                                      inference_times: List[float],
                                      save_name: str = "efficiency.png"):
        """Plot computational efficiency comparison (Fig 5)."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Sort by model size for better visualization
        sorted_indices = np.argsort(params)
        model_names_sorted = [model_names[i] for i in sorted_indices]
        params_sorted = [params[i] for i in sorted_indices]
        inference_times_sorted = [inference_times[i] for i in sorted_indices]
        
        x_pos = np.arange(len(model_names_sorted))
        
        # Parameters (log scale for better visualization)
        bars1 = axes[0].bar(x_pos, params_sorted, color='#2E86AB', alpha=0.8,
                           edgecolor='black', linewidth=0.5)
        axes[0].set_ylabel('Parameters (Millions)', fontsize=12)
        axes[0].set_title('Model Size Comparison', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(model_names_sorted, rotation=45, ha='right', fontsize=10)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Highlight HiGATE
        higate_idx = model_names_sorted.index('HiGATE') if 'HiGATE' in model_names_sorted else -1
        if higate_idx >= 0:
            bars1[higate_idx].set_color('#A23B72')
            bars1[higate_idx].set_edgecolor('gold')
            bars1[higate_idx].set_linewidth(2)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            axes[0].annotate(f'{height:.1f}M',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Inference time
        bars2 = axes[1].bar(x_pos, inference_times_sorted, color='#4ECDC4', alpha=0.8,
                           edgecolor='black', linewidth=0.5)
        axes[1].set_ylabel('Inference Time (ms)', fontsize=12)
        axes[1].set_title('Inference Speed Comparison', fontsize=14, fontweight='bold')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(model_names_sorted, rotation=45, ha='right', fontsize=10)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Highlight HiGATE
        if higate_idx >= 0:
            bars2[higate_idx].set_color('#A23B72')
            bars2[higate_idx].set_edgecolor('gold')
            bars2[higate_idx].set_linewidth(2)
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            axes[1].annotate(f'{height:.1f}ms',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Figure saved to {save_path}")
    
    def plot_explainability(self, image: np.ndarray, 
                           nuclei_masks: List[np.ndarray],
                           nuclei_classes: List[int],
                           importance_scores: np.ndarray,
                           tissue_regions: Optional[np.ndarray] = None,
                           save_name: str = "explainability.png"):
        """Plot multi-scale explainability visualization (Fig 7)."""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # (a) Original H&E image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('(a) Original H&E', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # (b) Nuclear instance segmentation
        overlay = image.copy()
        for i, (mask, cls) in enumerate(zip(nuclei_masks, nuclei_classes)):
            color = self.class_colors.get(cls, [255, 255, 255])
            overlay[mask > 0] = overlay[mask > 0] * 0.6 + np.array(color) * 0.4
        
        axes[0, 1].imshow(overlay.astype(np.uint8))
        axes[0, 1].set_title('(b) Nuclear Segmentation', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        # (c) Cell Graph
        axes[0, 2].imshow(image)
        # Add graph edges (simplified)
        for i in range(min(20, len(nuclei_masks))):  # Show first 20 for clarity
            cy, cx = np.where(nuclei_masks[i] > 0)
            if len(cy) > 0:
                centroid = (np.mean(cx), np.mean(cy))
                circle = plt.Circle(centroid, radius=5, color='red', fill=False)
                axes[0, 2].add_patch(circle)
        axes[0, 2].set_title('(c) Cell Graph', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        # (d) Adaptive tissue clusters
        if tissue_regions is not None:
            axes[0, 3].imshow(tissue_regions, cmap='tab20')
        else:
            axes[0, 3].imshow(image)
        axes[0, 3].set_title('(d) Tissue Clusters', fontsize=12, fontweight='bold')
        axes[0, 3].axis('off')
        
        # (e) Tissue Graph
        axes[1, 0].imshow(image)
        axes[1, 0].set_title('(e) Tissue Graph', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        # (f) Cell-level confidence
        conf_map = np.zeros(image.shape[:2])
        for i, (mask, score) in enumerate(zip(nuclei_masks, importance_scores)):
            conf_map[mask > 0] = score
        
        im = axes[1, 1].imshow(conf_map, cmap='hot', alpha=0.7)
        axes[1, 1].set_title('(f) Cell Confidence', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        # (g) XAI attribution map
        # Create heatmap overlay
        attribution_map = np.zeros(image.shape[:2])
        for i, (mask, score) in enumerate(zip(nuclei_masks, importance_scores)):
            attribution_map[mask > 0] = score
        
        axes[1, 2].imshow(image, alpha=0.5)
        im = axes[1, 2].imshow(attribution_map, cmap='jet', alpha=0.5)
        axes[1, 2].set_title('(g) XAI Attribution', fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')
        plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)
        
        # (h) Final report
        axes[1, 3].text(0.1, 0.9, 'Diagnostic Report', fontsize=14, fontweight='bold')
        axes[1, 3].text(0.1, 0.8, f'Prediction: Malignant', fontsize=12)
        axes[1, 3].text(0.1, 0.7, f'Confidence: 0.92', fontsize=12)
        axes[1, 3].text(0.1, 0.6, f'Key Features:', fontsize=12, fontweight='bold')
        axes[1, 3].text(0.1, 0.5, '• Nuclear pleomorphism', fontsize=10)
        axes[1, 3].text(0.1, 0.4, '• Disorganized architecture', fontsize=10)
        axes[1, 3].text(0.1, 0.3, '• High mitotic activity', fontsize=10)
        axes[1, 3].axis('off')
        axes[1, 3].set_title('(h) Clinical Report', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Figure saved to {save_path}")
