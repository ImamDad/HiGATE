
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class CrossLevelAttention(nn.Module):
    """
    Cross-Level Attention Mechanism (Section 3.4.2)
    Enables bidirectional message passing between cell and tissue hierarchies
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super(CrossLevelAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Bottom-up attention (cell → tissue)
        self.bottom_up_attn = MultiHeadAttention(hidden_dim, num_heads, 'bottom_up')
        
        # Top-down attention (tissue → cell)  
        self.top_down_attn = MultiHeadAttention(hidden_dim, num_heads, 'top_down')
        
        # Layer normalization
        self.cell_norm = nn.LayerNorm(hidden_dim)
        self.tissue_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, cell_features: torch.Tensor, tissue_features: torch.Tensor,
                cluster_labels: torch.Tensor, tissue_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-level attention between cell and tissue graphs
        
        Args:
            cell_features: [N_cell, hidden_dim] cell node features
            tissue_features: [N_tissue, hidden_dim] tissue node features  
            cluster_labels: [N_cell] cluster assignment for each cell
            tissue_batch: [N_tissue] batch assignment for tissue nodes
            
        Returns:
            Updated cell and tissue features
        """
        # Bottom-up attention: refine tissue features using cell information
        tissue_updated = self.bottom_up_attention(cell_features, tissue_features, cluster_labels)
        
        # Top-down attention: refine cell features using tissue context
        cell_updated = self.top_down_attention(cell_features, tissue_updated, cluster_labels)
        
        # Residual connections with layer norm
        cell_features = self.cell_norm(cell_features + cell_updated)
        tissue_features = self.tissue_norm(tissue_features + tissue_updated)
        
        return cell_features, tissue_features
    
    def bottom_up_attention(self, cell_features: torch.Tensor, tissue_features: torch.Tensor,
                          cluster_labels: torch.Tensor) -> torch.Tensor:
        """
        Bottom-up attention: tissue nodes attend to their constituent cells
        Implements equations (10)-(11) from paper
        """
        unique_clusters = torch.unique(cluster_labels[cluster_labels != -1])
        
        if len(unique_clusters) == 0:
            return tissue_features
        
        # Group cells by cluster
        cluster_cells = []
        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id
            cluster_cells.append(cell_features[mask])
        
        # Process each tissue node
        tissue_queries = tissue_features[unique_clusters]
        attended_tissues = []
        
        for i, cluster_id in enumerate(unique_clusters):
            cells_in_cluster = cluster_cells[i]
            if len(cells_in_cluster) > 0:
                # Tissue node as query, cells as keys/values
                tissue_query = tissue_queries[i].unsqueeze(0)  # [1, hidden_dim]
                cell_keys = cells_in_cluster.unsqueeze(0)      # [1, N_cells, hidden_dim]
                cell_values = cells_in_cluster.unsqueeze(0)    # [1, N_cells, hidden_dim]
                
                # Apply attention
                attended, _ = self.bottom_up_attn(
                    tissue_query, cell_keys, cell_values
                )
                attended_tissues.append(attended.squeeze(0))
            else:
                attended_tissues.append(tissue_queries[i])
        
        # Update tissue features
        tissue_updated = tissue_features.clone()
        tissue_updated[unique_clusters] = torch.stack(attended_tissues)
        
        return tissue_updated
    
    def top_down_attention(self, cell_features: torch.Tensor, tissue_features: torch.Tensor,
                         cluster_labels: torch.Tensor) -> torch.Tensor:
        """
        Top-down attention: cells attend to their tissue context  
        Implements equation (12) from paper
        """
        cell_queries = cell_features
        attended_cells = []
        
        for i, cell_feat in enumerate(cell_queries):
            cluster_id = cluster_labels[i]
            if cluster_id != -1:  # Skip noise points
                # Cell as query, tissue as key/value
                cell_query = cell_feat.unsqueeze(0)                    # [1, hidden_dim]
                tissue_key = tissue_features[cluster_id].unsqueeze(0)  # [1, hidden_dim]
                tissue_value = tissue_features[cluster_id].unsqueeze(0) # [1, hidden_dim]
                
                # Apply attention
                attended, _ = self.top_down_attn(cell_query, tissue_key, tissue_value)
                attended_cells.append(attended.squeeze(0))
            else:
                attended_cells.append(cell_feat)
        
        return torch.stack(attended_cells)


class MultiHeadAttention(nn.Module):
    """Multi-head attention implementation for cross-level interactions"""
    
    def __init__(self, hidden_dim: int, num_heads: int, attention_type: str):
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.attention_type = attention_type
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Scale factor
        self.scale = self.head_dim ** -0.5
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-head attention forward pass
        
        Args:
            query: [batch_size, seq_len, hidden_dim] or [seq_len, hidden_dim]
            key: [batch_size, seq_len, hidden_dim] or [seq_len, hidden_dim]  
            value: [batch_size, seq_len, hidden_dim] or [seq_len, hidden_dim]
            
        Returns:
            attended: [batch_size, seq_len, hidden_dim] or [seq_len, hidden_dim]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        # Ensure 3D tensors
        if query.dim() == 2:
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, seq_len, _ = query.size()
        
        # Linear projections
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        attended = self.out_proj(attended)
        
        if squeeze_output:
            attended = attended.squeeze(0)
            
        return attended, attention_weights


class DualAttentionFusion(nn.Module):
    """Dual attention fusion module (from your existing code)"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.local_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.global_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.attention = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, local_feats: torch.Tensor, global_feats: torch.Tensor) -> torch.Tensor:
        local_trans = self.local_proj(local_feats)
        global_trans = self.global_proj(global_feats)
        
        attention_input = torch.cat([local_trans, global_trans], dim=-1)
        attention_weights = self.attention(attention_input)
        
        fused_feats = (attention_weights[:, 0].unsqueeze(-1) * local_trans + 
                      attention_weights[:, 1].unsqueeze(-1) * global_trans)
        
        return self.out_proj(fused_feats) + local_feats
