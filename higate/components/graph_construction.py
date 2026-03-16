"""Hierarchical graph construction with learnable components."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
from typing import Tuple, Optional

class LearnableAdjacency(nn.Module):
    """Spatially-constrained learnable graph construction.
    
    Implements Equation 6:
    A_ij = σ(λ · sim(f_i, f_j) + (1-λ) · exp(-||p_i - p_j||² / 2σ_d²))
    """
    
    def __init__(self, feature_dim: int = 512, spatial_decay: float = 50.0):
        super().__init__()
        self.spatial_decay = spatial_decay
        self.lambda_weight = nn.Parameter(torch.tensor(0.5))
        
        # Optional: learn temperature for similarity
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, features: torch.Tensor, positions: torch.Tensor, 
                k: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct learnable adjacency matrix.
        
        Args:
            features: Node features of shape (N, feature_dim)
            positions: Spatial coordinates of shape (N, 2)
            k: Number of nearest neighbors to retain
            
        Returns:
            adj: Sparse adjacency matrix of shape (N, N)
            lambda_: Learned lambda value
        """
        N = features.size(0)
        
        # Normalize features for cosine similarity
        feat_norm = F.normalize(features, dim=1)
        sim_matrix = torch.mm(feat_norm, feat_norm.t())  # (N, N)
        
        # Compute pairwise distances
        dist_matrix = torch.cdist(positions, positions, p=2)  # (N, N)
        spatial_weight = torch.exp(-dist_matrix**2 / (2 * self.spatial_decay**2))
        
        # Learnable combination (Equation 6)
        lambda_ = torch.sigmoid(self.lambda_weight)
        adj = lambda_ * sim_matrix + (1 - lambda_) * spatial_weight
        adj = torch.sigmoid(adj * self.temperature)
        
        # Sparsify: keep top-k edges per node
        mask = torch.zeros_like(adj)
        topk_vals, topk_idx = torch.topk(adj, k=k, dim=1)
        mask.scatter_(1, topk_idx, 1.0)
        adj = adj * mask
        
        return adj, lambda_

class DifferentiablePooling(nn.Module):
    """Learnable tissue region formation via differentiable pooling.
    
    Implements Equations 7-9:
    S = softmax(GNN_pool(X, A))  (Equation 7)
    X' = S^T X, A' = S^T A S  (Equation 8)
    L_spatial = Σ Σ S_ik S_jk ||p_i - p_j||²  (Equation 9)
    """
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, 
                 max_clusters: int = 50):
        super().__init__()
        
        self.gnn_pool = nn.ModuleList([
            GCNConv(input_dim, hidden_dim),
            GCNConv(hidden_dim, max_clusters)
        ])
        
        self.max_clusters = max_clusters
        self.spatial_weight = nn.Parameter(torch.tensor(1.0))
        
    def _get_dynamic_clusters(self, N: int, batch: Optional[torch.Tensor] = None) -> int:
        """Dynamically determine number of clusters based on cell density."""
        if batch is None:
            K = max(5, min(self.max_clusters, int(np.ceil(N / 50))))
        else:
            unique_batches = batch.unique()
            avg_cells_per_batch = len(batch) / len(unique_batches)
            K = max(5, min(self.max_clusters, int(np.ceil(avg_cells_per_batch / 50))))
        
        return K
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                positions: torch.Tensor, batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate soft assignments and compute spatial loss.
        
        Args:
            x: Node features of shape (N, input_dim)
            edge_index: Graph connectivity of shape (2, E)
            positions: Spatial coordinates of shape (N, 2)
            batch: Batch assignment (optional)
            
        Returns:
            S: Soft assignment matrix of shape (N, K)
            L_spatial: Spatial regularization loss
        """
        N = x.size(0)
        
        # Generate assignment scores (Equation 7)
        s = F.relu(self.gnn_pool[0](x, edge_index))
        s = self.gnn_pool[1](s, edge_index)  # (N, max_clusters)
        
        # Get dynamic number of clusters
        K = self._get_dynamic_clusters(N, batch)
        s = s[:, :K]
        
        # Soft assignment
        S = F.softmax(s, dim=1)  # (N, K)
        
        # Compute spatial regularization loss (Equation 9)
        L_spatial = self._compute_spatial_loss(S, positions, edge_index)
        
        return S, L_spatial
    
    def _compute_spatial_loss(self, S: torch.Tensor, positions: torch.Tensor,
                              edge_index: torch.Tensor) -> torch.Tensor:
        """Compute spatial regularization term."""
        N, K = S.shape
        loss = 0.0
        num_edges = edge_index.size(1)
        
        if num_edges == 0:
            return torch.tensor(0.0, device=S.device)
        
        # For each edge
        for idx in range(num_edges):
            i, j = edge_index[0, idx], edge_index[1, idx]
            
            # Compute distance
            dist = torch.norm(positions[i] - positions[j])
            
            # For each cluster
            for k in range(K):
                if S[i, k] > 0 and S[j, k] > 0:
                    loss += S[i, k] * S[j, k] * dist**2
        
        return self.spatial_weight * loss / num_edges

class HierarchicalGraphBuilder(nn.Module):
    """End-to-end hierarchical graph construction (Algorithm 1)."""
    
    def __init__(self, feature_dim: int = 512, spatial_decay: float = 50.0):
        super().__init__()
        
        self.adj_learner = LearnableAdjacency(feature_dim, spatial_decay)
        self.pooling = DifferentiablePooling(feature_dim)
        
    def _dense_to_sparse(self, adj: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
        """Convert dense adjacency matrix to sparse edge_index."""
        mask = adj > threshold
        edge_index = mask.nonzero().t().contiguous()
        return edge_index
    
    def forward(self, features: torch.Tensor, positions: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> Tuple[Data, Data, torch.Tensor, torch.Tensor]:
        """Build hierarchical graphs.
        
        Args:
            features: Node features of shape (N, feature_dim)
            positions: Spatial coordinates of shape (N, 2)
            batch: Batch assignment (optional)
            
        Returns:
            cell_graph: Cell-level graph data
            tissue_graph: Tissue-level graph data
            S: Assignment matrix of shape (N, K)
            L_spatial: Spatial regularization loss
        """
        # Step 1: Build cell graph with learnable adjacency
        adj, lambda_ = self.adj_learner(features, positions)
        edge_index = self._dense_to_sparse(adj)
        
        cell_graph = Data(
            x=features,
            edge_index=edge_index,
            pos=positions,
            lambda_=lambda_
        )
        
        # Step 2: Learnable pooling with spatial regularization
        S, L_spatial = self.pooling(features, edge_index, positions, batch)
        
        # Step 3: Build tissue graph via pooling (Equation 8, 10)
        X_t = torch.mm(S.t(), features)  # (K, feature_dim)
        A_coarse = torch.mm(S.t(), torch.mm(adj, S))  # (K, K)
        
        # Threshold for sparsity (Equation 11)
        mask = A_coarse > 0.1
        A_t = A_coarse * mask.float()
        
        tissue_edge_index = self._dense_to_sparse(A_t)
        tissue_graph = Data(
            x=X_t,
            edge_index=tissue_edge_index,
            pos=None  # Region centroids could be computed if needed
        )
        
        return cell_graph, tissue_graph, S, L_spatial
