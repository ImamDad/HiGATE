"""Multi-modal feature extraction modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List, Tuple

class DINOv2FeatureExtractor(nn.Module):
    """Domain-adapted DINOv2 visual feature extractor.
    
    Implements Equation 2: f_i^vis = DINOv2_finetuned(ROI) ∈ R^768
    """
    
    def __init__(self, pretrained: bool = True, fine_tune: bool = True,
                 output_dim: int = 256):
        super().__init__()
        
        # Load DINOv2 ViT-B/14 backbone
        self.backbone = timm.create_model(
            'vit_base_patch14_dinov2.lvd142m',
            pretrained=pretrained,
            num_classes=0
        )
        self.backbone_dim = 768
        
        # Projection head
        if fine_tune:
            self.projection = nn.Sequential(
                nn.Linear(self.backbone_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, output_dim)
            )
        else:
            self.projection = nn.Identity()
        
        self.output_dim = output_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract visual features.
        
        Args:
            x: Input tensor of shape (B, 3, 224, 224)
            
        Returns:
            Visual features of shape (B, output_dim)
        """
        features = self.backbone(x)  # (B, 768)
        features = self.projection(features)  # (B, output_dim)
        
        return features

class MorphologicalFeatureExtractor(nn.Module):
    """Extract morphological features from nuclear masks.
    
    Implements Equation 3: f_i^morph = [area, perimeter, eccentricity, solidity, extent, orientation]
    """
    
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, output_dim: int = 128):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process morphological features.
        
        Args:
            x: Morphological features of shape (B, 6)
            
        Returns:
            Embedded features of shape (B, output_dim)
        """
        return self.mlp(x)

class StarDistFeatureExtractor(nn.Module):
    """Extract fine-grained nuclear features from StarDist."""
    
    def __init__(self, input_dim: int = 12, hidden_dim: int = 64, output_dim: int = 128):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process StarDist features.
        
        Args:
            x: StarDist features of shape (B, 12)
            
        Returns:
            Embedded features of shape (B, output_dim)
        """
        return self.mlp(x)

class AttentionFusion(nn.Module):
    """Attention-weighted multi-modal feature fusion.
    
    Implements Equations 4-5:
    f_i = sum(α_m · W_m f_i^m)  (Equation 4)
    α_m = exp(v^T tanh(U f_i^m)) / sum(exp(v^T tanh(U f_i^n)))  (Equation 5)
    """
    
    def __init__(self, dims: List[int] = [256, 128, 128], 
                 hidden_dim: int = 128, output_dim: int = 512):
        super().__init__()
        
        # Modality-specific projections
        self.projections = nn.ModuleList([
            nn.Linear(d, output_dim) for d in dims
        ])
        
        # Shared attention parameters
        self.U = nn.Linear(output_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, features_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse multi-modal features with attention.
        
        Args:
            features_list: List of [vis_feats, morph_feats, nuc_feats]
                          each of shape (B, d_i)
                          
        Returns:
            fused: Fused features of shape (B, output_dim)
            attn_weights: Attention weights of shape (B, 3)
        """
        B = features_list[0].size(0)
        
        # Project all modalities to same dimension
        projected = []
        for proj, feat in zip(self.projections, features_list):
            projected.append(proj(feat))
        
        # Stack projected features
        stacked = torch.stack(projected, dim=1)  # (B, 3, output_dim)
        
        # Compute attention scores (Equation 5)
        attn_scores = []
        for i in range(3):
            score = self.v(torch.tanh(self.U(stacked[:, i, :])))  # (B, 1)
            attn_scores.append(score)
        
        attn_weights = torch.softmax(torch.cat(attn_scores, dim=1), dim=1)  # (B, 3)
        
        # Weighted fusion (Equation 4)
        fused = torch.sum(attn_weights.unsqueeze(-1) * stacked, dim=1)  # (B, output_dim)
        fused = self.norm(fused)
        
        return fused, attn_weights
