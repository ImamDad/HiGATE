"""Loss functions for HiGATE training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

class FocalLoss(nn.Module):
    """Focal loss with class and tissue weighting.
    
    Implements Equation 1:
    L = -∑ w_c · w_t(i) · (1 - p_i,c)^γ log(p_i,c)
    """
    
    def __init__(self, class_weights: Optional[List[float]] = None,
                 tissue_weights: Optional[List[float]] = None,
                 gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights))
        else:
            self.class_weights = None
            
        if tissue_weights is not None:
            self.register_buffer('tissue_weights', torch.tensor(tissue_weights))
        else:
            self.tissue_weights = None
            
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                tissue_types: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            logits: Model predictions of shape (B, C) or (1, C)
            targets: Ground truth labels of shape (B,) or (1,)
            tissue_types: Tissue type indices for each sample
            
        Returns:
            Focal loss value
        """
        # Handle single sample case
        if logits.dim() == 2 and logits.size(0) == 1:
            logits = logits.squeeze(0)
            
        # Compute log softmax
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        
        # Gather probabilities of target classes
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal weights
        focal_weights = (1 - target_probs) ** self.gamma
        
        # Apply class weights
        if self.class_weights is not None:
            class_weights = self.class_weights[targets]
            focal_weights = focal_weights * class_weights
        
        # Apply tissue weights
        if self.tissue_weights is not None and tissue_types is not None:
            tissue_weights = self.tissue_weights[tissue_types]
            focal_weights = focal_weights * tissue_weights
        
        # Compute loss
        loss = -focal_weights * log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class SpatialRegularizationLoss(nn.Module):
    """Spatial regularization loss for graph pooling."""
    
    def __init__(self, weight: float = 0.01):
        super().__init__()
        self.weight = weight
        
    def forward(self, S: torch.Tensor, positions: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """Compute spatial regularization loss."""
        N, K = S.shape
        loss = 0.0
        num_edges = edge_index.size(1)
        
        if num_edges == 0:
            return torch.tensor(0.0, device=S.device)
        
        for idx in range(num_edges):
            i, j = edge_index[0, idx], edge_index[1, idx]
            dist = torch.norm(positions[i] - positions[j])
            
            for k in range(K):
                if S[i, k] > 0 and S[j, k] > 0:
                    loss += S[i, k] * S[j, k] * dist**2
        
        return self.weight * loss / num_edges

class HiGATELoss(nn.Module):
    """Combined loss for HiGATE training."""
    
    def __init__(self, config):
        super().__init__()
        
        self.focal_loss = FocalLoss(
            class_weights=config.class_weights,
            tissue_weights=config.tissue_weights,
            gamma=config.focal_gamma
        )
        
        self.spatial_loss = SpatialRegularizationLoss(
            weight=config.spatial_weight
        )
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                S: torch.Tensor, positions: torch.Tensor,
                edge_index: torch.Tensor,
                tissue_types: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute total loss."""
        # Classification loss
        cls_loss = self.focal_loss(logits, targets, tissue_types)
        
        # Spatial regularization loss
        spatial_loss = self.spatial_loss(S, positions, edge_index)
        
        # Total loss
        total_loss = cls_loss + spatial_loss
        
        return {
            'total': total_loss,
            'classification': cls_loss,
            'spatial': spatial_loss
        }
