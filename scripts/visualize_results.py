import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import logging
from config import config

logger = logging.getLogger(__name__)

class ResultsVisualizer:
    """Visualization tools for HiGATE results"""
    
    def __init__(self, results_dir: Path = config.RESULTS_PATH):
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_training_curves(self, train_metrics: dict, val_metrics: dict, save_path: Path = None):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(train_metrics.get('loss', []), label='Train Loss', linewidth=2)
        axes[0, 0].plot(val_metrics.get('loss', []), label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(train_metrics.get('accuracy', []), label='Train Accuracy', linewidth=2)
        axes[0, 1].plot(val_metrics.get('accuracy', []), label='Val Accuracy', linewidth=2)
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1-score curves
        axes[1, 0].plot(train_metrics.get('f1_score', []), label='Train F1-Score', linewidth=2)
        axes[1, 0].plot(val_metrics.get('f1_score', []), label='Val F1-Score', linewidth=2)
        axes[1, 0].set_title('Training and Validation F1-Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate
        if 'learning_rate' in train_metrics:
            axes[1, 1].plot(train_metrics['learning_rate'], label='Learning Rate', linewidth=2)
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved training curves to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: list, save_path: Path = None):
        """Plot confusion matrix"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Add labels
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               title='Confusion Matrix',
               ylabel='True Label',
               xlabel='Predicted Label')
        
        # Rotate tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        thresh = cm_normalized.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]}\n({cm_normalized[i, j]:.2f})",
                       ha="center", va="center",
                       color="white" if cm_normalized[i, j] > thresh else "black")
        
        fig.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")
        
        plt.show()
    
    def plot_per_class_metrics(self, metrics: dict, save_path: Path = None):
        """Plot per-class performance metrics"""
        classes = config.CLASS_NAMES
        metrics_df = pd.DataFrame(metrics).T
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy per class
        if 'accuracy' in metrics_df.columns:
            axes[0, 0].bar(classes, metrics_df['accuracy'])
            axes[0, 0].set_title('Accuracy per Class')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # F1-score per class
        if 'f1_score' in metrics_df.columns:
            axes[0, 1].bar(classes, metrics_df['f1_score'])
            axes[0, 1].set_title('F1-Score per Class')
            axes[0, 1].set_ylabel('F1-Score')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Precision per class
        if 'precision' in metrics_df.columns:
            axes[1, 0].bar(classes, metrics_df['precision'])
            axes[1, 0].set_title('Precision per Class')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Recall per class
        if 'recall' in metrics_df.columns:
            axes[1, 1].bar(classes, metrics_df['recall'])
            axes[1, 1].set_title('Recall per Class')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved per-class metrics to {save_path}")
        
        plt.show()

def main():
    """Example usage of visualizer"""
    visualizer = ResultsVisualizer()
    
    # Example data (replace with actual results)
    train_metrics = {
        'loss': [0.8, 0.6, 0.5, 0.4, 0.3],
        'accuracy': [0.7, 0.75, 0.8, 0.82, 0.85],
        'f1_score': [0.68, 0.73, 0.78, 0.80, 0.83]
    }
    
    val_metrics = {
        'loss': [0.75, 0.65, 0.55, 0.45, 0.35],
        'accuracy': [0.65, 0.70, 0.75, 0.78, 0.80],
        'f1_score': [0.63, 0.68, 0.73, 0.76, 0.78]
    }
    
    # Plot training curves
    visualizer.plot_training_curves(train_metrics, val_metrics)

if __name__ == "__main__":
    main()
