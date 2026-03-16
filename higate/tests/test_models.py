"""Unit tests for HiGATE model components."""

import unittest
import torch
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.components.feature_extractors import (
    DINOv2FeatureExtractor,
    MorphologicalFeatureExtractor,
    StarDistFeatureExtractor,
    AttentionFusion
)
from models.components.graph_construction import (
    LearnableAdjacency,
    DifferentiablePooling,
    HierarchicalGraphBuilder
)
from models.components.attention_mechanisms import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    BidirectionalCrossLevelAttention
)
from models.components.explainability import (
    IntegratedGradients,
    LayerwiseRelevancePropagation,
    PerturbationAnalyzer
)
from models.higate import HiGATE, GATLayer
from config.default_config import ModelConfig, HiGATEConfig


class TestFeatureExtractors(unittest.TestCase):
    """Test feature extraction modules."""
    
    def setUp(self):
        self.batch_size = 4
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def test_dinov2_extractor(self):
        """Test DINOv2 feature extractor."""
        model = DINOv2FeatureExtractor(pretrained=False, fine_tune=True, output_dim=256)
        model = model.to(self.device)
        
        # Create dummy input
        x = torch.randn(self.batch_size, 3, 224, 224).to(self.device)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 256))
        
    def test_morphological_extractor(self):
        """Test morphological feature extractor."""
        model = MorphologicalFeatureExtractor(input_dim=6, output_dim=128)
        model = model.to(self.device)
        
        # Create dummy input
        x = torch.randn(self.batch_size, 6).to(self.device)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 128))
        
    def test_stardist_extractor(self):
        """Test StarDist feature extractor."""
        model = StarDistFeatureExtractor(input_dim=12, output_dim=128)
        model = model.to(self.device)
        
        # Create dummy input
        x = torch.randn(self.batch_size, 12).to(self.device)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 128))
        
    def test_attention_fusion(self):
        """Test attention-weighted fusion module."""
        model = AttentionFusion(dims=[256, 128, 128], output_dim=512)
        model = model.to(self.device)
        
        # Create dummy inputs
        vis_feats = torch.randn(self.batch_size, 256).to(self.device)
        morph_feats = torch.randn(self.batch_size, 128).to(self.device)
        nuc_feats = torch.randn(self.batch_size, 128).to(self.device)
        
        # Forward pass
        fused, attn_weights = model([vis_feats, morph_feats, nuc_feats])
        
        # Check output shapes
        self.assertEqual(fused.shape, (self.batch_size, 512))
        self.assertEqual(attn_weights.shape, (self.batch_size, 3))
        
        # Check attention weights sum to 1
        self.assertTrue(torch.allclose(attn_weights.sum(dim=1), torch.ones(self.batch_size).to(self.device)))


class TestGraphConstruction(unittest.TestCase):
    """Test graph construction modules."""
    
    def setUp(self):
        self.num_nodes = 100
        self.feature_dim = 512
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def test_learnable_adjacency(self):
        """Test learnable adjacency construction."""
        model = LearnableAdjacency(feature_dim=self.feature_dim, spatial_decay=50.0)
        model = model.to(self.device)
        
        # Create dummy inputs
        features = torch.randn(self.num_nodes, self.feature_dim).to(self.device)
        positions = torch.randn(self.num_nodes, 2).to(self.device)
        
        # Forward pass
        adj, lambda_ = model(features, positions, k=20)
        
        # Check output shapes
        self.assertEqual(adj.shape, (self.num_nodes, self.num_nodes))
        self.assertIsInstance(lambda_, torch.Tensor)
        
        # Check sparsity (top-20 edges per node)
        self.assertEqual((adj > 0).sum(dim=1).max(), 20)
        
    def test_differentiable_pooling(self):
        """Test differentiable pooling module."""
        model = DifferentiablePooling(input_dim=self.feature_dim, max_clusters=50)
        model = model.to(self.device)
        
        # Create dummy inputs
        features = torch.randn(self.num_nodes, self.feature_dim).to(self.device)
        positions = torch.randn(self.num_nodes, 2).to(self.device)
        
        # Create random edge_index
        edge_index = torch.randint(0, self.num_nodes, (2, 200)).to(self.device)
        
        # Forward pass
        S, L_spatial = model(features, edge_index, positions)
        
        # Check output shapes
        self.assertEqual(S.shape[0], self.num_nodes)
        self.assertLessEqual(S.shape[1], 50)
        self.assertGreaterEqual(S.shape[1], 5)
        self.assertIsInstance(L_spatial, torch.Tensor)
        
        # Check softmax properties
        self.assertTrue(torch.allclose(S.sum(dim=1), torch.ones(self.num_nodes).to(self.device), atol=1e-5))
        
    def test_hierarchical_graph_builder(self):
        """Test complete hierarchical graph builder."""
        model = HierarchicalGraphBuilder(feature_dim=self.feature_dim)
        model = model.to(self.device)
        
        # Create dummy inputs
        features = torch.randn(self.num_nodes, self.feature_dim).to(self.device)
        positions = torch.randn(self.num_nodes, 2).to(self.device)
        
        # Forward pass
        cell_graph, tissue_graph, S, L_spatial = model(features, positions)
        
        # Check outputs
        self.assertEqual(cell_graph.x.shape, (self.num_nodes, self.feature_dim))
        self.assertEqual(cell_graph.edge_index.shape[0], 2)
        self.assertEqual(tissue_graph.x.shape[0], S.shape[1])
        self.assertEqual(tissue_graph.x.shape[1], self.feature_dim)
        self.assertEqual(S.shape[0], self.num_nodes)


class TestAttentionMechanisms(unittest.TestCase):
    """Test attention mechanisms."""
    
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 10
        self.d_model = 512
        self.n_heads = 4
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def test_scaled_dot_product_attention(self):
        """Test scaled dot-product attention."""
        d_k = self.d_model // self.n_heads
        model = ScaledDotProductAttention(d_k)
        model = model.to(self.device)
        
        # Create dummy inputs
        Q = torch.randn(self.batch_size, self.n_heads, self.seq_len, d_k).to(self.device)
        K = torch.randn(self.batch_size, self.n_heads, self.seq_len, d_k).to(self.device)
        V = torch.randn(self.batch_size, self.n_heads, self.seq_len, d_k).to(self.device)
        
        # Forward pass
        output, attn_weights = model(Q, K, V)
        
        # Check shapes
        self.assertEqual(output.shape, (self.batch_size, self.n_heads, self.seq_len, d_k))
        self.assertEqual(attn_weights.shape, (self.batch_size, self.n_heads, self.seq_len, self.seq_len))
        
    def test_multi_head_attention(self):
        """Test multi-head attention."""
        model = MultiHeadAttention(d_model=self.d_model, n_heads=self.n_heads)
        model = model.to(self.device)
        
        # Create dummy inputs
        Q = torch.randn(self.batch_size, self.seq_len, self.d_model).to(self.device)
        K = torch.randn(self.batch_size, self.seq_len, self.d_model).to(self.device)
        V = torch.randn(self.batch_size, self.seq_len, self.d_model).to(self.device)
        
        # Forward pass
        output, attn_weights = model(Q, K, V)
        
        # Check shapes
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        
    def test_bidirectional_cross_level_attention(self):
        """Test bidirectional cross-level attention."""
        model = BidirectionalCrossLevelAttention(d_model=self.d_model, n_heads=self.n_heads)
        model = model.to(self.device)
        
        N, K = 100, 20  # Number of cells and tissue regions
        
        # Create dummy inputs
        h_cell = torch.randn(N, self.d_model).to(self.device)
        h_tissue = torch.randn(K, self.d_model).to(self.device)
        S = torch.softmax(torch.randn(N, K), dim=1).to(self.device)
        
        # Forward pass
        h_cell_out, h_tissue_out = model(h_cell, h_tissue, S)
        
        # Check shapes
        self.assertEqual(h_cell_out.shape, (N, self.d_model))
        self.assertEqual(h_tissue_out.shape, (K, self.d_model))


class TestGATLayer(unittest.TestCase):
    """Test GAT layer."""
    
    def setUp(self):
        self.num_nodes = 100
        self.in_dim = 512
        self.out_dim = 512
        self.n_heads = 4
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def test_gat_layer(self):
        """Test GAT layer forward pass."""
        model = GATLayer(self.in_dim, self.out_dim, self.n_heads)
        model = model.to(self.device)
        
        # Create dummy inputs
        x = torch.randn(self.num_nodes, self.in_dim).to(self.device)
        edge_index = torch.randint(0, self.num_nodes, (2, 200)).to(self.device)
        
        # Forward pass
        output = model(x, edge_index)
        
        # Check shape
        self.assertEqual(output.shape, (self.num_nodes, self.out_dim))


class TestHiGATE(unittest.TestCase):
    """Test complete HiGATE model."""
    
    def setUp(self):
        self.config = ModelConfig()
        self.config.num_classes = 5
        self.config.num_layers = 2  # Use fewer layers for testing
        self.config.fused_dim = 512
        
        self.batch_size = 4
        self.num_cells = 50
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create HiGATE model
        self.model = HiGATE(self.config)
        self.model = self.model.to(self.device)
        
    def test_forward_pass(self):
        """Test model forward pass."""
        # Create dummy inputs
        images = torch.randn(self.num_cells, 3, 224, 224).to(self.device)
        morph_features = torch.randn(self.num_cells, 6).to(self.device)
        stardist_features = torch.randn(self.num_cells, 12).to(self.device)
        positions = torch.randn(self.num_cells, 2).to(self.device)
        
        # Forward pass
        output = self.model(images, morph_features, stardist_features, positions)
        
        # Check output keys
        expected_keys = ['logits', 'attn_weights', 'S', 'L_spatial', 'cell_attn', 'tissue_attn', 'z_cell', 'z_tissue']
        for key in expected_keys:
            self.assertIn(key, output)
        
        # Check shapes
        self.assertEqual(output['logits'].shape, (1, self.config.num_classes))
        self.assertEqual(output['attn_weights'].shape, (self.num_cells, 3))
        self.assertEqual(output['S'].shape[0], self.num_cells)
        self.assertIsInstance(output['L_spatial'], torch.Tensor)
        self.assertEqual(output['z_cell'].shape, (self.config.fused_dim,))
        self.assertEqual(output['z_tissue'].shape, (self.config.fused_dim,))
        
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        # Create dummy inputs with gradients
        images = torch.randn(self.num_cells, 3, 224, 224, requires_grad=True).to(self.device)
        morph_features = torch.randn(self.num_cells, 6, requires_grad=True).to(self.device)
        stardist_features = torch.randn(self.num_cells, 12, requires_grad=True).to(self.device)
        positions = torch.randn(self.num_cells, 2).to(self.device)
        
        # Forward pass
        output = self.model(images, morph_features, stardist_features, positions)
        
        # Backward pass
        loss = output['logits'].sum()
        loss.backward()
        
        # Check gradients
        self.assertIsNotNone(images.grad)
        self.assertIsNotNone(morph_features.grad)
        self.assertIsNotNone(stardist_features.grad)
        
    def test_parameter_count(self):
        """Test that model has reasonable number of parameters."""
        num_params = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(num_params, 0)
        self.assertLess(num_params, 10_000_000)  # Should be less than 10M


class TestExplainability(unittest.TestCase):
    """Test explainability modules."""
    
    def setUp(self):
        self.config = ModelConfig()
        self.config.num_classes = 5
        self.config.num_layers = 1
        
        self.num_cells = 20
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.model = HiGATE(self.config)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Create dummy inputs
        self.images = torch.randn(self.num_cells, 3, 224, 224).to(self.device)
        self.morph_features = torch.randn(self.num_cells, 6).to(self.device)
        self.stardist_features = torch.randn(self.num_cells, 12).to(self.device)
        self.positions = torch.randn(self.num_cells, 2).to(self.device)
        
    def test_integrated_gradients(self):
        """Test Integrated Gradients explainability."""
        ig = IntegratedGradients(self.model, steps=10)
        
        # Compute explanations
        importance = ig.get_node_importance(
            self.images, self.morph_features, self.stardist_features, self.positions
        )
        
        # Check shape
        self.assertEqual(importance.shape[0], self.num_cells)
        
    def test_perturbation_analyzer(self):
        """Test perturbation analysis."""
        analyzer = PerturbationAnalyzer(self.model)
        
        # Create importance scores
        importance_scores = torch.randn(self.num_cells).to(self.device)
        
        # Analyze
        results = analyzer.analyze(
            self.images, self.morph_features, self.stardist_features,
            self.positions, importance_scores, k=5
        )
        
        # Check results keys
        expected_keys = ['baseline_confidence', 'sufficiency', 'comprehensiveness']
        for key in expected_keys:
            self.assertIn(key, results)


if __name__ == '__main__':
    unittest.main()
