import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_max_pool
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import Delaunay
import logging

logger = logging.getLogger(__name__)

class HiGATE(nn.Module):
    """
    HiGATE: Hierarchical Graph Attention with Cross-Level Interaction
    Implementation exactly matching the research paper
    """
    
    def __init__(self, config):
        super(HiGATE, self).__init__()
        self.config = config
        
        # Feature projection for adaptive clustering (Section 3.3.2)
        self.feature_projection = nn.Sequential(
            nn.Linear(config.CELL_FEATURE_DIM, 392),
            nn.BatchNorm1d(392),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(392, 196),
            nn.BatchNorm1d(196),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(196, 64),
            nn.BatchNorm1d(64),
            nn.Linear(64, config.CLUSTERING_DIM)  # Linear projection for clustering
        )
        
        # Input projection
        self.cell_input_proj = nn.Linear(config.CELL_FEATURE_DIM, config.HIDDEN_DIM)
        self.tissue_input_proj = nn.Linear(config.CELL_FEATURE_DIM, config.HIDDEN_DIM)
        
        # Cell-level GAT layers (Section 3.4.1)
        self.cell_gnn_layers = nn.ModuleList([
            GATConv(config.HIDDEN_DIM, config.HIDDEN_DIM // config.NUM_HEADS, 
                   heads=config.NUM_HEADS, dropout=config.DROPOUT_RATE)
            for _ in range(config.NUM_LAYERS)
        ])
        
        # Tissue-level GCN layers (Section 3.4.1)
        self.tissue_gnn_layers = nn.ModuleList([
            GCNConv(config.HIDDEN_DIM, config.HIDDEN_DIM)
            for _ in range(config.NUM_LAYERS)
        ])
        
        # Cross-Level Attention (Section 3.4.2)
        self.cross_attention = CrossLevelAttention(
            hidden_dim=config.HIDDEN_DIM,
            num_heads=config.NUM_HEADS
        )
        
        # Top-down GRU (Section 3.4.2)
        self.top_down_gru = nn.GRUCell(config.HIDDEN_DIM * 2, config.HIDDEN_DIM)
        
        # Classification head (Section 3.5)
        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM * 4, config.HIDDEN_DIM * 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_DIM * 2, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_DIM, config.NUM_CLASSES)
        )
        
    def forward(self, data):
        """
        Forward pass exactly following paper methodology
        """
        # Extract input data
        cell_features = data['cell_features']
        centroids = data['centroids']
        cell_edge_index = data['cell_edge_index']
        batch = data.get('batch', None)
        
        # Feature projection for clustering (Eq. in Section 3.3.2)
        projected_features = self.feature_projection(cell_features)
        
        # Adaptive tissue graph construction (Algorithm 1)
        tissue_graph, cluster_labels = self._construct_tissue_graph(
            projected_features, centroids, cell_features, batch
        )
        
        # Initialize representations
        cell_h = self.cell_input_proj(cell_features)
        tissue_h = self.tissue_input_proj(tissue_graph.x)
        
        # Multi-scale graph learning with cross-level attention
        for layer_idx in range(self.config.NUM_LAYERS):
            # Intra-level processing
            cell_h = F.elu(self.cell_gnn_layers[layer_idx](cell_h, cell_edge_index))
            tissue_h = F.relu(self.tissue_gnn_layers[layer_idx](tissue_h, tissue_graph.edge_index))
            
            # Cross-level attention (Section 3.4.2)
            cell_h, tissue_h = self.cross_attention(
                cell_h, tissue_h, cluster_labels, tissue_graph.batch
            )
            
            # Top-down refinement with GRU (Eq. 12)
            cell_h = self._top_down_refinement(cell_h, tissue_h, cluster_labels)
        
        # Multi-scale readout (Section 3.5)
        cell_embedding = self._readout_cell_graph(cell_h, batch)
        tissue_embedding = self._readout_tissue_graph(tissue_h, tissue_graph.batch)
        
        # Final classification
        combined_embedding = torch.cat([cell_embedding, tissue_embedding], dim=-1)
        logits = self.classifier(combined_embedding)
        
        return logits
    
    def _construct_tissue_graph(self, projected_features, centroids, original_features, batch):
        """Construct tissue graph using adaptive DBSCAN clustering (Section 3.3.3)"""
        
        # Move to CPU for clustering
        projected_np = projected_features.detach().cpu().numpy()
        centroids_np = centroids.detach().cpu().numpy()
        
        # Adaptive parameter selection (Section 3.3.3)
        eps = self._compute_adaptive_eps(projected_np)
        min_samples = max(3, int(0.01 * len(projected_np)))  # δ = 0.01 as in paper
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(projected_np)
        labels = clustering.labels_
        
        # Create tissue nodes
        unique_labels = np.unique(labels[labels != -1])
        tissue_nodes = []
        tissue_positions = []
        
        for cluster_id in unique_labels:
            mask = labels == cluster_id
            if np.sum(mask) > 0:
                # Mean-pool features for tissue node (Eq. for tissue feature aggregation)
                cluster_features = original_features[mask].mean(dim=0)
                cluster_centroid = centroids[mask].mean(dim=0)
                tissue_nodes.append(cluster_features)
                tissue_positions.append(cluster_centroid)
        
        if len(tissue_nodes) == 0:
            # Fallback: use global average
            tissue_nodes = [original_features.mean(dim=0)]
            tissue_positions = [centroids.mean(dim=0)]
            labels = np.zeros(len(projected_np))
        
        tissue_nodes = torch.stack(tissue_nodes)
        tissue_positions = torch.stack(tissue_positions)
        
        # Build tissue graph edges using Delaunay triangulation
        if len(tissue_positions) > 3:
            try:
                tri = Delaunay(tissue_positions.cpu().numpy())
                edges = set()
                for simplex in tri.simplices:
                    for i in range(3):
                        for j in range(i+1, 3):
                            edges.add((simplex[i], simplex[j]))
                edge_index = torch.tensor(list(edges)).t().contiguous()
            except:
                # Fallback: complete graph
                edge_index = self._create_complete_graph(len(tissue_positions))
        else:
            edge_index = self._create_complete_graph(len(tissue_positions))
        
        # Create tissue graph data structure
        from torch_geometric.data import Data
        tissue_graph = Data(
            x=tissue_nodes,
            edge_index=edge_index,
            pos=tissue_positions,
            batch=torch.zeros(len(tissue_nodes), dtype=torch.long)
        )
        
        return tissue_graph, labels
    
    def _compute_adaptive_eps(self, features):
        """Compute adaptive epsilon for DBSCAN (Section 3.3.3)"""
        from sklearn.neighbors import NearestNeighbors
        n_samples = len(features)
        n_neighbors = min(5, n_samples - 1)
        
        if n_neighbors <= 0:
            return 0.5
            
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(features)
        distances, _ = nbrs.kneighbors(features)
        k_distances = distances[:, -1]
        eps = np.percentile(k_distances, 95)  # 95th percentile as in paper
        return float(eps)
    
    def _create_complete_graph(self, n_nodes):
        """Create complete graph for small number of nodes"""
        edges = []
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                edges.append([i, j])
        return torch.tensor(edges).t().contiguous() if edges else torch.zeros(2, 0, dtype=torch.long)
    
    def _top_down_refinement(self, cell_features, tissue_features, cluster_labels):
        """Top-down refinement using GRU (Section 3.4.2)"""
        refined_cell_features = []
        
        for i, cell_feat in enumerate(cell_features):
            cluster_id = cluster_labels[i]
            if cluster_id != -1:  # Skip noise points
                tissue_feat = tissue_features[cluster_id]
                # GRU update as in paper
                combined_input = torch.cat([cell_feat.unsqueeze(0), tissue_feat.unsqueeze(0)], dim=-1)
                refined_feat = self.top_down_gru(combined_input, cell_feat.unsqueeze(0))
                refined_cell_features.append(refined_feat.squeeze(0))
            else:
                refined_cell_features.append(cell_feat)
        
        return torch.stack(refined_cell_features)
    
    def _readout_cell_graph(self, features, batch):
        """Dual-pooling readout for cell graph (Eq. 13)"""
        if batch is None:
            batch = torch.zeros(features.size(0), dtype=torch.long, device=features.device)
        
        mean_pool = global_mean_pool(features, batch)
        max_pool = global_max_pool(features, batch)
        return torch.cat([mean_pool, max_pool], dim=-1)
    
    def _readout_tissue_graph(self, features, batch):
        """Dual-pooling readout for tissue graph (Eq. 14)"""
        if batch is None:
            batch = torch.zeros(features.size(0), dtype=torch.long, device=features.device)
            
        mean_pool = global_mean_pool(features, batch)
        max_pool = global_max_pool(features, batch)
        return torch.cat([mean_pool, max_pool], dim=-1)


class CrossLevelAttention(nn.Module):
    """Cross-Level Attention Mechanism (Section 3.4.2)"""
    
    def __init__(self, hidden_dim, num_heads):
        super(CrossLevelAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Bottom-up attention (cell-to-tissue)
        self.bottom_up_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        
        # Top-down attention (tissue-to-cell)  
        self.top_down_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        
    def forward(self, cell_features, tissue_features, cluster_labels, tissue_batch):
        # Bottom-up attention: tissue queries, cell keys/values
        tissue_features = self._bottom_up_attention(
            tissue_features, cell_features, cluster_labels
        )
        
        # Top-down attention: cell queries, tissue keys/values  
        cell_features = self._top_down_attention(
            cell_features, tissue_features, cluster_labels
        )
        
        return cell_features, tissue_features
    
    def _bottom_up_attention(self, tissue_features, cell_features, cluster_labels):
        """Bottom-up attention: tissue nodes attend to their constituent cells"""
        unique_clusters = torch.unique(cluster_labels[cluster_labels != -1])
        
        updated_tissue_features = []
        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id
            cluster_cells = cell_features[mask]
            
            if len(cluster_cells) > 0:
                # Tissue node as query, cells as keys/values
                tissue_query = tissue_features[cluster_id].unsqueeze(0).unsqueeze(0)
                cell_keys = cluster_cells.unsqueeze(0)
                cell_values = cluster_cells.unsqueeze(0)
                
                attended, _ = self.bottom_up_attn(tissue_query, cell_keys, cell_values)
                updated_tissue_features.append(attended.squeeze(0))
            else:
                updated_tissue_features.append(tissue_features[cluster_id])
        
        if updated_tissue_features:
            tissue_features[unique_clusters] = torch.stack(updated_tissue_features)
        
        return tissue_features
    
    def _top_down_attention(self, cell_features, tissue_features, cluster_labels):
        """Top-down attention: cells attend to their tissue context"""
        updated_cell_features = []
        
        for i, cell_feat in enumerate(cell_features):
            cluster_id = cluster_labels[i]
            if cluster_id != -1:
                # Cell as query, tissue as key/value
                cell_query = cell_feat.unsqueeze(0).unsqueeze(0)
                tissue_key = tissue_features[cluster_id].unsqueeze(0).unsqueeze(0)
                tissue_value = tissue_features[cluster_id].unsqueeze(0).unsqueeze(0)
                
                attended, _ = self.top_down_attn(cell_query, tissue_key, tissue_value)
                updated_cell_features.append(attended.squeeze(0))
            else:
                updated_cell_features.append(cell_feat)
        
        return torch.stack(updated_cell_features)
