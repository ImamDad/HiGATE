"""Bidirectional cross-level attention mechanisms."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention."""
    
    def __init__(self, d_k: int):
        super().__init__()
        self.d_k = d_k
        self.scale = 1.0 / math.sqrt(d_k)
        
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention.
        
        Args:
            Q: Query of shape (..., seq_len_q, d_k)
            K: Key of shape (..., seq_len_k, d_k)
            V: Value of shape (..., seq_len_k, d_v)
            mask: Optional mask
            
        Returns:
            output: Attended values
            attn_weights: Attention weights
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights

class MultiHeadAttention(nn.Module):
    """Multi-head attention module."""
    
    def __init__(self, d_model: int = 512, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        
        # Attention
        self.attention = ScaledDotProductAttention(self.d_k)
        
        # Dropout and normalization
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Multi-head attention forward pass."""
        batch_size = Q.size(0)
        
        # Linear projections and split into heads
        Q = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        # Final projection
        output = self.fc(attn_output)
        output = self.dropout(output)
        output = self.norm(output)
        
        return output, attn_weights

class BidirectionalCrossLevelAttention(nn.Module):
    """Symmetric bidirectional cross-level attention mechanism.
    
    Implements Equations 13-16:
    - Bottom-up: cells → tissue (Equation 13)
    - Top-down: tissue → cells (Equation 14)
    - Iterative refinement (Equations 15-16)
    """
    
    def __init__(self, d_model: int = 512, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        # Bottom-up attention (cells -> tissue)
        self.bottom_up_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Top-down attention (tissue -> cells)
        self.top_down_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Projection layers for bottom-up
        self.W_bu_q = nn.Linear(d_model, d_model)
        self.W_bu_k = nn.Linear(d_model, d_model)
        self.W_bu_v = nn.Linear(d_model, d_model)
        
        # Projection layers for top-down
        self.W_td_q = nn.Linear(d_model, d_model)
        self.W_td_k = nn.Linear(d_model, d_model)
        self.W_td_v = nn.Linear(d_model, d_model)
        
        # Gating mechanisms
        self.gate_bu = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        self.gate_td = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
    def forward(self, h_cell: torch.Tensor, h_tissue: torch.Tensor, 
                S: torch.Tensor, k_indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply bidirectional cross-level attention.
        
        Args:
            h_cell: Cell representations of shape (N, d_model)
            h_tissue: Tissue region representations of shape (K, d_model)
            S: Soft assignment matrix of shape (N, K)
            k_indices: Primary tissue assignment for each cell (optional)
            
        Returns:
            h_cell_updated: Refined cell representations
            h_tissue_updated: Refined tissue representations
        """
        N, K = S.shape
        d = h_cell.size(1)
        
        # Get primary assignments if not provided
        if k_indices is None:
            k_indices = torch.argmax(S, dim=1)  # (N,)
        
        # ===== Bottom-up: cells -> tissue (Equation 13) =====
        h_tissue_updated = h_tissue.clone()
        
        # For each tissue region, attend to its constituent cells
        for k in range(K):
            # Get cells belonging to this region
            cell_mask = S[:, k] > 0.1
            if cell_mask.sum() == 0:
                continue
            
            cells_k = h_cell[cell_mask].unsqueeze(0)  # (1, M, d)
            region_k = h_tissue[k].unsqueeze(0).unsqueeze(1)  # (1, 1, d)
            
            # Project
            Q = self.W_bu_q(region_k)  # (1, 1, d)
            K_proj = self.W_bu_k(cells_k)  # (1, M, d)
            V = self.W_bu_v(cells_k)  # (1, M, d)
            
            # Attend
            attn_out, _ = self.bottom_up_attn(Q, K_proj, V)
            
            # Gated update
            gate = self.gate_bu(torch.cat([
                h_tissue[k].unsqueeze(0).unsqueeze(0),
                attn_out
            ], dim=-1))
            h_tissue_updated[k] = (gate.squeeze() * attn_out.squeeze() + 
                                   (1 - gate.squeeze()) * h_tissue[k])
        
        # ===== Top-down: tissue -> cells (Equation 14) =====
        # Gather tissue regions for each cell
        tissue_for_cells = h_tissue_updated[k_indices]  # (N, d)
        
        # Project
        Q = self.W_td_q(h_cell.unsqueeze(1))  # (N, 1, d)
        K_proj = self.W_td_k(tissue_for_cells.unsqueeze(1))  # (N, 1, d)
        V = self.W_td_v(tissue_for_cells.unsqueeze(1))  # (N, 1, d)
        
        # Attend
        attn_out, _ = self.top_down_attn(Q, K_proj, V)  # (N, 1, d)
        attn_out = attn_out.squeeze(1)  # (N, d)
        
        # Gated update
        gate = self.gate_td(torch.cat([h_cell, attn_out], dim=-1))
        h_cell_updated = gate * attn_out + (1 - gate) * h_cell
        
        return h_cell_updated, h_tissue_updated
