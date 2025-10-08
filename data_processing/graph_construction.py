import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.spatial import Delaunay
from typing import Tuple, Dict
import logging
from config import config

logger = logging.getLogger(__name__)

class HierarchicalGraphBuilder:
    """
    Hierarchical graph construction exactly as described in paper Section 3.3
    Builds Cell Graph (k-NN) and Tissue Graph (adaptive DBSCAN)
    """
    
    def __init__(self, config):
        self.config = config
        
    def build_cell_graph(self, features: Dict[str, torch.Tensor]) -> Data:
        """
        Build Cell Graph using adaptive k-NN (Section 3.3.1)
        """
        combined_features = features['combined_features']
        centroids = features['centroids']
        
        if len(combined_features) == 0:
            return self._create_empty_graph()
        
        # Adaptive k-NN connectivity
        k = self._compute_adaptive_k(len(centroids))
        edge_index = self._build_knn_edges(centroids, k)
        
        return Data(
            x=combined_features,
            edge_index=edge_index,
            pos=centroids,
            num_nodes=len(combined_features)
        )
    
    def build_tissue_graph(self, features: Dict[str, torch.Tensor]) -> Tuple[Data, np.ndarray]:
        """
        Build Tissue Graph using adaptive DBSCAN clustering (Section 3.3.3)
        Returns tissue graph and cluster labels
        """
        combined_features = features['combined_features']
        centroids = features['centroids']
        
        if len(combined_features) < 2:
            return self._create_empty_graph(), np.array([])
        
        # Feature projection for clustering
        projected_features = self._project_features(combined_features)
        
        # Adaptive DBSCAN clustering
        cluster_labels = self._adaptive_dbscan_clustering(projected_features, centroids)
        
        # Build tissue graph from clusters
        tissue_graph = self._build_tissue_graph_from_clusters(
            combined_features, centroids, cluster_labels
        )
        
        return tissue_graph, cluster_labels
    
    def _compute_adaptive_k(self, n_nodes: int) -> int:
        """Compute adaptive k for k-NN (Section 3.3.1)"""
        k = min(self.config.K_MAX, max(self.config.K_MIN, int(np.sqrt(n_nodes))))
        return k
    
    def _build_knn_edges(self, centroids: torch.Tensor, k: int) -> torch.Tensor:
        """Build k-NN edges between cell centroids"""
        if len(centroids) <= 1:
            return torch.zeros(2, 0, dtype=torch.long)
        
        centroids_np = centroids.cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=k).fit(centroids_np)
        distances, indices = nbrs.kneighbors(centroids_np)
        
        edges = []
        for i in range(len(centroids)):
            for j in indices[i]:
                if i != j:  # Avoid self-loops
                    edges.append([i, j])
        
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    def _project_features(self, features: torch.Tensor) -> np.ndarray:
        """Project features for clustering (simplified version)"""
        # In full implementation, this would use the learned projection from HiGATE
        return features.detach().cpu().numpy()
    
    def _adaptive_dbscan_clustering(self, features: np.ndarray, centroids: torch.Tensor) -> np.ndarray:
        """Adaptive DBSCAN clustering (Section 3.3.3)"""
        n_samples = len(features)
        if n_samples < 2:
            return np.zeros(n_samples, dtype=int)
        
        # Compute adaptive parameters
        eps = self._compute_adaptive_eps(features)
        min_samples = max(self.config.MIN_CLUSTER_SIZE, int(0.01 * n_samples))
        
        # Apply DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
        return clustering.labels_
    
    def _compute_adaptive_eps(self, features: np.ndarray) -> float:
        """Compute adaptive epsilon for DBSCAN"""
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
    
    def _build_tissue_graph_from_clusters(self, features: torch.Tensor, 
                                        centroids: torch.Tensor, 
                                        cluster_labels: np.ndarray) -> Data:
        """Build tissue graph from clusters using Delaunay triangulation"""
        unique_labels = np.unique(cluster_labels[cluster_labels != -1])
        
        if len(unique_labels) == 0:
            return self._create_empty_graph()
        
        # Create tissue nodes (mean-pooled features and centroids)
        tissue_features = []
        tissue_positions = []
        
        for cluster_id in unique_labels:
            mask = cluster_labels == cluster_id
            if np.sum(mask) > 0:
                cluster_feats = features[mask].mean(dim=0)
                cluster_center = centroids[mask].mean(dim=0)
                tissue_features.append(cluster_feats)
                tissue_positions.append(cluster_center)
        
        tissue_features = torch.stack(tissue_features)
        tissue_positions = torch.stack(tissue_positions)
        
        # Build edges using Delaunay triangulation
        if len(tissue_positions) > 3:
            try:
                edge_index = self._delaunay_edges(tissue_positions)
            except:
                edge_index = self._complete_graph_edges(len(tissue_positions))
        else:
            edge_index = self._complete_graph_edges(len(tissue_positions))
        
        return Data(
            x=tissue_features,
            edge_index=edge_index,
            pos=tissue_positions,
            num_nodes=len(tissue_features)
        )
    
    def _delaunay_edges(self, positions: torch.Tensor) -> torch.Tensor:
        """Build edges using Delaunay triangulation"""
        positions_np = positions.cpu().numpy()
        tri = Delaunay(positions_np)
        
        edges = set()
        for simplex in tri.simplices:
            for i in range(len(simplex)):
                for j in range(i+1, len(simplex)):
                    edges.add((simplex[i], simplex[j]))
        
        edge_list = list(edges)
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.zeros(2, 0, dtype=torch.long)
    
    def _complete_graph_edges(self, n_nodes: int) -> torch.Tensor:
        """Build complete graph for small number of nodes"""
        edges = []
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                edges.append([i, j])
        return torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros(2, 0, dtype=torch.long)
    
    def _create_empty_graph(self) -> Data:
        """Create empty graph"""
        return Data(
            x=torch.zeros(0, self.config.CELL_FEATURE_DIM),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            pos=torch.zeros(0, 2),
            num_nodes=0
        )
