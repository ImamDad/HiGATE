"""Main HiGATE model implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from typing import Dict, Optional, Tuple

from .components.feature_extractors import (
    DINOv2FeatureExtractor,
    MorphologicalFeatureExtractor,
    StarDistFeatureExtractor,
    AttentionFusion
)
from .components.graph_construction import HierarchicalGraphBuilder
from .components.attention_mechanisms import BidirectionalCrossLevelAttention

class GATLayer(nn.Module):
    """Graph Attention Network layer.
    
    Implements Equation 12:
    h_i^(l+1) = || σ(∑ α_ij^(l),h W^(l),h h_j^(l))
    """
    
    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4, 
                 dropout: float = 0.2):
        super().__init__()
        
        self.gat = GATConv(
            in_dim, out_dim // n_heads, 
            heads=n_heads, concat=True, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Apply GAT layer."""
        x = self.gat(x, edge_index)
        x = self.dropout(x)
        x = self.norm(x)
        return F.elu(x)

class HiGATE(nn.Module):
    """Hierarchical Graph Attention Tissue Encoder.
    
    Main model integrating all components:
    - Multi-modal feature extraction
    - Hierarchical graph construction
    - Bidirectional cross-level attention
    - Multi-scale readout and classification
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # ===== Feature Extraction =====
        self.dinov2_extractor = DINOv2FeatureExtractor(
            pretrained=True,
            fine_tune=config.fine_tune_dinov2,
            output_dim=config.visual_dim
        )
        
        self.morph_extractor = MorphologicalFeatureExtractor(
            input_dim=6,
            output_dim=config.morph_dim
        )
        
        self.stardist_extractor = StarDistFeatureExtractor(
            input_dim=12,
            output_dim=config.nuclear_dim
        )
        
        # Multi-modal fusion
        self.fusion = AttentionFusion(
            dims=[config.visual_dim, config.morph_dim, config.nuclear_dim],
            output_dim=config.fused_dim
        )
        
        # ===== Graph Construction =====
        self.graph_builder = HierarchicalGraphBuilder(
            feature_dim=config.fused_dim,
            spatial_decay=config.spatial_decay
        )
        
        # ===== Intra-level GAT Layers =====
        self.cell_gat_layers = nn.ModuleList([
            GATLayer(config.fused_dim, config.fused_dim, 
                    config.num_heads, config.dropout)
            for _ in range(config.num_layers)
        ])
        
        self.tissue_gat_layers = nn.ModuleList([
            GATLayer(config.fused_dim, config.fused_dim,
                    config.num_heads, config.dropout)
            for _ in range(config.num_layers)
        ])
        
        # ===== Cross-level Attention =====
        self.cross_attention = BidirectionalCrossLevelAttention(
            d_model=config.fused_dim,
            n_heads=config.num_heads,
            dropout=config.dropout
        )
        
        # ===== Readout Layers =====
        # Attention-based pooling (Equations 17-18)
        self.cell_readout = nn.Sequential(
            nn.Linear(config.fused_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1, bias=False)
        )
        
        self.tissue_readout = nn.Sequential(
            nn.Linear(config.fused_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1, bias=False)
        )
        
        # ===== Classifier =====
        self.classifier = nn.Sequential(
            nn.Linear(config.fused_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, config.num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, images: torch.Tensor, morph_features: torch.Tensor,
                stardist_features: torch.Tensor, positions: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass of HiGATE.
        
        Args:
            images: ROI images of shape (N, 3, 224, 224)
            morph_features: Morphological features of shape (N, 6)
            stardist_features: StarDist features of shape (N, 12)
            positions: Spatial coordinates of shape (N, 2)
            batch: Batch assignment (optional)
            
        Returns:
            Dictionary containing:
                logits: Classification logits of shape (1, num_classes)
                attn_weights: Fusion attention weights of shape (N, 3)
                S: Assignment matrix of shape (N, K)
                L_spatial: Spatial regularization loss
                cell_attn: Cell attention weights
                tissue_attn: Tissue attention weights
        """
        # Step 1: Multi-modal feature extraction
        vis_feats = self.dinov2_extractor(images)          # (N, visual_dim)
        morph_feats = self.morph_extractor(morph_features)  # (N, morph_dim)
        nuc_feats = self.stardist_extractor(stardist_features)  # (N, nuclear_dim)
        
        # Step 2: Attention-weighted fusion (Equations 4-5)
        fused_feats, attn_weights = self.fusion(
            [vis_feats, morph_feats, nuc_feats]
        )  # (N, fused_dim)
        
        # Step 3: Hierarchical graph construction (Algorithm 1)
        cell_graph, tissue_graph, S, L_spatial = self.graph_builder(
            fused_feats, positions, batch
        )
        
        # Get primary tissue assignment for each cell
        k_indices = torch.argmax(S, dim=1)
        
        # Initialize representations
        h_cell = cell_graph.x
        h_tissue = tissue_graph.x
        
        # Step 4: Iterative refinement with bidirectional attention
        # (Equations 15-16)
        for layer in range(self.config.num_layers):
            # Intra-level processing
            h_cell = self.cell_gat_layers[layer](h_cell, cell_graph.edge_index)
            h_tissue = self.tissue_gat_layers[layer](h_tissue, tissue_graph.edge_index)
            
            # Cross-level bidirectional attention
            h_cell, h_tissue = self.cross_attention(
                h_cell, h_tissue, S, k_indices
            )
        
        # Step 5: Readout and classification
        
        # Attention pooling for cell graph (Equation 17)
        cell_scores = self.cell_readout(h_cell)  # (N, 1)
        cell_attn = F.softmax(cell_scores, dim=0)  # (N, 1)
        z_cell = torch.sum(cell_attn * h_cell, dim=0)  # (fused_dim,)
        
        # Attention pooling for tissue graph (Equation 18)
        tissue_scores = self.tissue_readout(h_tissue)  # (K, 1)
        tissue_attn = F.softmax(tissue_scores, dim=0)  # (K, 1)
        z_tissue = torch.sum(tissue_attn * h_tissue, dim=0)  # (fused_dim,)
        
        # Concatenate for final representation (Equation 19)
        z = torch.cat([z_cell, z_tissue], dim=0).unsqueeze(0)  # (1, fused_dim*2)
        
        # Classification
        logits = self.classifier(z)  # (1, num_classes)
        
        return {
            'logits': logits,
            'attn_weights': attn_weights,
            'S': S,
            'L_spatial': L_spatial,
            'cell_attn': cell_attn.squeeze(),
            'tissue_attn': tissue_attn.squeeze(),
            'z_cell': z_cell,
            'z_tissue': z_tissue
        }
