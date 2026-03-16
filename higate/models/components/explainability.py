"""Explainability modules for HiGATE."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Optional, Dict, List
import numpy as np

class IntegratedGradients:
    """Integrated Gradients for node importance.
    
    Implements Equation 20:
    IG_i = (h_i - h_i^baseline) × ∫ ∂f/∂h_i dα
    """
    
    def __init__(self, model: nn.Module, steps: int = 50):
        self.model = model
        self.steps = steps
        
    def explain(self, images: torch.Tensor, morph_features: torch.Tensor,
                stardist_features: torch.Tensor, positions: torch.Tensor,
                target_class: Optional[int] = None) -> torch.Tensor:
        """Compute integrated gradients for each node.
        
        Args:
            images: ROI images of shape (N, 3, 224, 224)
            morph_features: Morphological features of shape (N, 6)
            stardist_features: StarDist features of shape (N, 12)
            positions: Spatial coordinates of shape (N, 2)
            target_class: Target class for attribution
            
        Returns:
            Integrated gradients of shape (total_features,)
        """
        self.model.eval()
        
        # Baselines (zero features)
        baseline_images = torch.zeros_like(images)
        baseline_morph = torch.zeros_like(morph_features)
        baseline_stardist = torch.zeros_like(stardist_features)
        baseline_positions = positions.clone()
        
        # Create scaled inputs along linear path
        scaled_inputs = []
        alphas = torch.linspace(0, 1, self.steps)
        
        for alpha in alphas:
            scaled_images = baseline_images + alpha * (images - baseline_images)
            scaled_morph = baseline_morph + alpha * (morph_features - baseline_morph)
            scaled_stardist = baseline_stardist + alpha * (stardist_features - baseline_stardist)
            scaled_inputs.append((scaled_images, scaled_morph, scaled_stardist))
        
        # Accumulate gradients
        total_grad = None
        
        for scaled_images, scaled_morph, scaled_stardist in scaled_inputs:
            scaled_images.requires_grad_(True)
            scaled_morph.requires_grad_(True)
            scaled_stardist.requires_grad_(True)
            
            # Forward pass
            output = self.model(
                scaled_images, scaled_morph, scaled_stardist, baseline_positions
            )
            
            # Get target class
            if target_class is None:
                target_class = output['logits'].argmax(dim=1).item()
            
            # Compute gradients
            self.model.zero_grad()
            output['logits'][0, target_class].backward()
            
            # Concatenate gradients
            grad = torch.cat([
                scaled_images.grad.view(-1),
                scaled_morph.grad.view(-1),
                scaled_stardist.grad.view(-1)
            ])
            
            if total_grad is None:
                total_grad = grad
            else:
                total_grad += grad
        
        # Compute integrated gradients (Equation 20)
        avg_grad = total_grad / self.steps
        baseline_concat = torch.cat([
            baseline_images.view(-1),
            baseline_morph.view(-1),
            baseline_stardist.view(-1)
        ])
        input_concat = torch.cat([
            images.view(-1),
            morph_features.view(-1),
            stardist_features.view(-1)
        ])
        
        ig = (input_concat - baseline_concat) * avg_grad
        
        return ig
    
    def get_node_importance(self, images: torch.Tensor, morph_features: torch.Tensor,
                           stardist_features: torch.Tensor, positions: torch.Tensor,
                           target_class: Optional[int] = None) -> torch.Tensor:
        """Get importance scores for each node."""
        ig = self.explain(images, morph_features, stardist_features, positions, target_class)
        
        # Reshape to per-node importance
        N = images.size(0)
        feat_per_node = (3 * 224 * 224) + 6 + 12
        node_importance = []
        
        for i in range(N):
            start = i * feat_per_node
            end = (i + 1) * feat_per_node
            node_imp = ig[start:end].abs().mean()
            node_importance.append(node_imp)
        
        return torch.tensor(node_importance)

class LayerwiseRelevancePropagation:
    """Layer-wise Relevance Propagation.
    
    Implements Equation 21:
    R_i^(l) = ∑ (z_ij / (∑ z_kj + ε)) R_j^(l+1)
    with conservation: ∑ R_i^(l) = ∑ R_j^(l+1)
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.relevance_maps: Dict[int, tuple] = {}
        self.hooks = []
        
    def _register_hooks(self):
        """Register forward hooks to capture activations."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, GATConv)):
                hook = module.register_forward_hook(self._forward_hook)
                self.hooks.append(hook)
    
    def _forward_hook(self, module, input, output):
        """Store forward pass for relevance propagation."""
        self.relevance_maps[id(module)] = (input[0].detach(), output.detach())
    
    def _remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.relevance_maps.clear()
    
    def _lrp_epsilon(self, module: nn.Module, x: torch.Tensor, y: torch.Tensor,
                     R_y: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
        """LRP epsilon rule."""
        if isinstance(module, nn.Linear):
            W = module.weight
            b = module.bias if module.bias is not None else 0
            
            # Compute contributions
            z = torch.mm(x, W.t()) + b
            s = R_y / (z + epsilon * torch.sign(z))
            R_x = torch.mm(s, W)
            
        elif isinstance(module, GATConv):
            # For GAT, use attention weights
            if hasattr(module, '_alpha'):
                attn = module._alpha
                R_x = torch.mm(R_y, attn.mean(dim=0))
            else:
                # Fallback: uniform redistribution
                R_x = torch.ones_like(x) * (R_y.sum() / x.size(0))
                
        elif isinstance(module, nn.Conv2d):
            # Simplified for conv layers
            R_x = torch.ones_like(x) * (R_y.sum() / x.numel())
            
        else:
            # Default: redistribute equally
            R_x = x * (R_y.sum() / x.numel())
        
        return R_x
    
    def explain(self, images: torch.Tensor, morph_features: torch.Tensor,
                stardist_features: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Compute relevance scores using LRP.
        
        Args:
            images: ROI images of shape (N, 3, 224, 224)
            morph_features: Morphological features of shape (N, 6)
            stardist_features: StarDist features of shape (N, 12)
            positions: Spatial coordinates of shape (N, 2)
            
        Returns:
            Relevance scores at input level
        """
        self.model.eval()
        self._register_hooks()
        
        # Forward pass
        with torch.no_grad():
            output = self.model(images, morph_features, stardist_features, positions)
        
        target = output['logits'].argmax(dim=1).item()
        
        # Initialize relevance at output
        R = torch.zeros_like(output['logits'])
        R[0, target] = output['logits'][0, target]
        
        # Propagate backwards through modules in reverse order
        modules = list(self.model.modules())
        for module in reversed(modules):
            if id(module) in self.relevance_maps:
                x, y = self.relevance_maps[id(module)]
                R = self._lrp_epsilon(module, x, y, R)
        
        self._remove_hooks()
        return R

class PerturbationAnalyzer:
    """Perturbation analysis for explanation validation."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        
    def analyze(self, images: torch.Tensor, morph_features: torch.Tensor,
               stardist_features: torch.Tensor, positions: torch.Tensor,
               importance_scores: torch.Tensor, k: int = 10) -> Dict[str, float]:
        """Analyze effect of removing important nodes.
        
        Args:
            images: ROI images
            morph_features: Morphological features
            stardist_features: StarDist features
            positions: Spatial coordinates
            importance_scores: Node importance scores
            k: Number of top nodes to remove
            
        Returns:
            Dictionary with sufficiency and comprehensiveness metrics
        """
        self.model.eval()
        
        # Get top-k important nodes
        N = images.size(0)
        top_k_idx = importance_scores.argsort(descending=True)[:k]
        
        # Baseline prediction
        with torch.no_grad():
            baseline_output = self.model(images, morph_features, stardist_features, positions)
        baseline_prob = F.softmax(baseline_output['logits'], dim=1)[0]
        
        # Sufficiency: use only top-k important nodes
        images_suff = images.clone()
        morph_suff = morph_features.clone()
        stardist_suff = stardist_features.clone()
        
        # Mask out other nodes
        mask = torch.zeros(N, dtype=torch.bool)
        mask[top_k_idx] = True
        
        images_suff[~mask] = 0
        morph_suff[~mask] = 0
        stardist_suff[~mask] = 0
        
        with torch.no_grad():
            suff_output = self.model(images_suff, morph_suff, stardist_suff, positions)
        suff_prob = F.softmax(suff_output['logits'], dim=1)[0]
        
        # Comprehensiveness: remove top-k important nodes
        images_comp = images.clone()
        morph_comp = morph_features.clone()
        stardist_comp = stardist_features.clone()
        
        images_comp[mask] = 0
        morph_comp[mask] = 0
        stardist_comp[mask] = 0
        
        with torch.no_grad():
            comp_output = self.model(images_comp, morph_comp, stardist_comp, positions)
        comp_prob = F.softmax(comp_output['logits'], dim=1)[0]
        
        return {
            'baseline_confidence': baseline_prob.max().item(),
            'sufficiency': (suff_prob.max() / baseline_prob.max()).item(),
            'comprehensiveness': (1 - comp_prob.max() / baseline_prob.max()).item()
        }
