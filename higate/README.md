I'll rewrite the README.md file to be more comprehensive, visually appealing, and professional for your GitHub repository.

```markdown
# 🧬 HiGATE: Hierarchical Graph Attention for Multi-Scale Tissue Encoder in Computational Pathology

<div align="center">

[![Paper](https://img.shields.io/badge/📄%20Paper-Journal%20of%20Translational%20Medicine-blue?style=for-the-badge)](https://doi.org/xxxx)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/ImamDad/HiGATE?style=for-the-badge&logo=github)](https://github.com/ImamDad/HiGATE)
[![GitHub Issues](https://img.shields.io/github/issues/ImamDad/HiGATE?style=for-the-badge&logo=github)](https://github.com/ImamDad/HiGATE/issues)

**Official PyTorch Implementation | State-of-the-Art in Computational Pathology**

[🏆 Key Features](#-key-features) • [📊 Results](#-results) • [🚀 Quick Start](#-quick-start) • [📖 Documentation](#-documentation) • [📝 Citation](#-citation)

</div>

---

https://github.com/user-attachments/assets/5d3f9c2a-1b8e-4d7a-9f6c-3e2a1d5b8f7a

## 📋 Abstract

**HiGATE (Hierarchical Graph Attention Tissue Encoder)** is a biologically-inspired framework that unifies cellular and tissue-level analysis through a novel dual-graph architecture with bidirectional cross-level attention. Unlike prior hierarchical models that rely on static, unidirectional information flow, HiGATE introduces dynamic, context-aware communication where cellular details inform tissue organization and architectural context refines cellular representations.

### Key Innovations

| 🚀 Innovation | 📝 Description | 📈 Impact |
|--------------|----------------|-----------|
| **Bidirectional Cross-Level Attention** | Dynamic information exchange between cellular and tissue scales | +3.4% accuracy improvement |
| **Learnable Spatially-Constrained Graph Construction** | Adaptive, biologically-meaningful region formation via differentiable pooling | +4.5% over fixed methods |
| **Multi-Modal Feature Fusion** | Integration of DINOv2, morphological descriptors, and StarDist features | +2.2% over concatenation |
| **Inherent Explainability** | Multi-scale explanations with Integrated Gradients and LRP | 4.1/5.0 pathologist rating |

## 🏆 State-of-the-Art Performance

HiGATE achieves **state-of-the-art performance** across multiple benchmarks:

<div align="center">

| Dataset | Task | Metric | HiGATE | HACT-Net | Improvement | p-value |
|---------|------|--------|--------|----------|-------------|---------|
| **PanNuke** | Nuclei Classification | Accuracy | **91.3%** | 89.1% | +2.2% | <0.001 |
| **PanNuke** | Nuclei Classification | F1-Score | **0.896** | 0.879 | +1.7% | <0.001 |
| **PanNuke** | Nuclei Classification | AUROC | **0.958** | 0.945 | +1.3% | <0.001 |
| **MoNuSeg** | Segmentation | Dice | **0.841** | 0.796 | +4.5% | <0.001 |
| **DigestPath** | Classification | Accuracy | **87.2%** | 83.2% | +4.0% | <0.001 |
| **TCGA-BRCA** | WSI Grading | Accuracy | **85.4%** | 83.5% | +1.9% | <0.05 |

</div>

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input WSI/Patch                           │
└───────────────────────────────┬─────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Multi-Modal Feature Extraction                │
├───────────────┬────────────────┬───────────────────────────────┤
│   DINOv2      │  Morphological │          StarDist              │
│   (256-dim)   │   (128-dim)    │         (128-dim)              │
└───────────────┴────────────────┴───────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│              Attention-Weighted Fusion (Eq. 4-5)                 │
│                    ┌─────────────────────┐                       │
│                    │   Fused Features    │                       │
│                    │     (512-dim)       │                       │
│                    └─────────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│              Cell Graph Construction (Eq. 6)                     │
│         Learnable Adjacency: A_ij = σ(λ·sim + (1-λ)·exp(-d²))    │
│                      k-NN sparsification (k=20)                   │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│           Differentiable Pooling with Spatial Reg. (Eq. 7-9)     │
│              S = softmax(GNN_pool(X, A))                          │
│              L_spatial = Σ Σ S_ik S_jk ||p_i - p_j||²            │
└─────────────────────────────────────────────────────────────────┘
              ↓                                  ↓
┌─────────────────────────┐          ┌─────────────────────────┐
│    Cell Graph GAT       │          │   Tissue Graph GAT      │
│    ┌───────────────┐    │          │   ┌───────────────┐     │
│    │   Layer 1     │    │          │   │   Layer 1     │     │
│    ├───────────────┤    │          │   ├───────────────┤     │
│    │   Layer 2     │    │          │   │   Layer 2     │     │
│    ├───────────────┤    │          │   ├───────────────┤     │
│    │   Layer 3     │    │          │   │   Layer 3     │     │
│    └───────────────┘    │          │   └───────────────┘     │
└───────────┬─────────────┘          └───────────┬─────────────┘
            └──────────────────┬──────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│            Bidirectional Cross-Level Attention (Eq. 13-16)       │
│                                                                    │
│   ┌─────────────────────────────────────────────────────┐        │
│   │  Bottom-Up: h_tissue ← Attn(h_tissue, h_cell)      │        │
│   │  Top-Down:  h_cell   ← Attn(h_cell, h_tissue)      │        │
│   └─────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
              ↓                                  ↓
┌─────────────────────────┐          ┌─────────────────────────┐
│   Cell Attention Pool   │          │  Tissue Attention Pool  │
│      (Eq. 17)           │          │       (Eq. 18)          │
│   z_c = Σ α_i h_i       │          │   z_t = Σ β_k h_k       │
└───────────┬─────────────┘          └───────────┬─────────────┘
            └──────────────────┬──────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│              Final Representation (Eq. 19)                       │
│                    z = [z_c || z_t]                               │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Classification Head                          │
│                    MLP → Softmax                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 🔬 Key Components

### 1. Multi-Modal Feature Extraction

| Modality | Source | Dimensions | Description |
|----------|--------|------------|-------------|
| **Visual** | DINOv2 ViT-B/14 | 256 | Domain-adapted visual semantics from 224×224 ROIs |
| **Morphological** | Geometric descriptors | 6 → 128 | Area, perimeter, eccentricity, solidity, extent, orientation |
| **Nuclear** | StarDist | 12 → 128 | Convexity, diameter, axis ratio, texture entropy, etc. |

**Equation 2-3:** `f_i_vis = DINOv2_finetuned(ROI)`, `f_i_morph = [area, perimeter, ...]`

### 2. Attention-Weighted Feature Fusion

**Equation 4:** `f_i = Σ α_m · W_m f_i^m`

**Equation 5:** `α_m = exp(v^T tanh(U f_i^m)) / Σ exp(v^T tanh(U f_i^n))`

### 3. Learnable Graph Construction

**Equation 6:** `A_ij = σ(λ · sim(f_i, f_j) + (1-λ) · exp(-||p_i - p_j||² / 2σ_d²))`

- λ is **learnable** (converges to 0.65 ± 0.08)
- σ_d = 50μm (approximately 5 cell diameters)
- Top-20 edges retained per node

### 4. Differentiable Pooling with Spatial Regularization

**Equation 7:** `S = softmax(GNN_pool(X, A))`

**Equation 8:** `X' = S^T X`, `A' = S^T A S`

**Equation 9:** `L_spatial = Σ Σ S_ik S_jk ||p_i - p_j||²`

- K = ⌈N/50⌉ dynamically determined
- Spatial coherence enforced anatomically

### 5. Bidirectional Cross-Level Attention

**Equation 13 (Bottom-Up):** `h̃_tissue = Σ S_ik · Attn(W_Q^bu h_tissue, W_K^bu h_cell, W_V^bu h_cell)`

**Equation 14 (Top-Down):** `h̃_cell = Attn(W_Q_td h_cell, W_K_td h_tissue, W_V_td h_tissue)`

**Equations 15-16 (Iterative Refinement):**
- `H_c^(l+1) = GAT(H_c^(l)) + TopDownAttn(H_t^(l), H_c^(l))`
- `H_t^(l+1) = GAT(H_t^(l)) + BottomUpAttn(H_c^(l), H_t^(l))`

### 6. Explainability Framework

| Method | Equation | Purpose |
|--------|----------|---------|
| **Integrated Gradients** | `IG_i = (h_i - h_i^base) × ∫ ∂f/∂h_i dα` | Node importance attribution |
| **Layer-wise Relevance Propagation** | `R_i^(l) = Σ (z_ij / (Σ z_kj + ε)) R_j^(l+1)` | Relevance conservation |
| **Perturbation Analysis** | Sufficiency & Comprehensiveness | Validation of attributions |

## 📈 Comprehensive Results

### Figure 2: ROC and Precision-Recall Curves

<div align="center">
<img src="figures/fig2a_roc_curves.png" width="45%"> <img src="figures/fig2b_pr_curves.png" width="45%">
</div>

HiGATE achieves **AUROC of 0.958** and **AUPRC of 0.901**, significantly outperforming:
- HACT-Net: 0.945 AUROC, 0.872 AUPRC
- Swin Transformer: 0.933 AUROC, 0.841 AUPRC
- TransPath: 0.938 AUROC, 0.849 AUPRC

### Figure 3: Training Dynamics

<div align="center">
<img src="figures/fig3_training_dynamics.png" width="80%">
</div>

- **2.1× faster convergence** than HACT-Net
- **2.3% generalization gap** (vs. 4.8% for HACT-Net)
- Reaches 90% accuracy by epoch 25

### Figure 4: Per-Class Accuracy

<div align="center">
<img src="figures/fig4_per_class_accuracy.png" width="80%">
</div>

Largest improvements in architecturally complex tissues:
- **Bile-Duct:** +4.7%
- **Bladder:** +4.2%
- **Kidney:** +3.9%
- **Prostate:** +3.5%

### Figure 5: Computational Efficiency

<div align="center">
<img src="figures/fig5_efficiency.png" width="80%">
</div>

| Model | Parameters | Inference Time | Memory |
|-------|------------|----------------|--------|
| **HiGATE** | **3.2M** | **20.9ms** | **1.9GB** |
| HACT-Net | 28.4M | 15.9ms | 2.1GB |
| Swin Transformer | 49.2M | 22.4ms | 3.1GB |
| TransPath | 87.2M | 24.1ms | 3.8GB |

### Figure 6: Ablation Study

<div align="center">
<img src="figures/fig6_ablation.png" width="70%">
</div>

| Component | Accuracy Drop | Significance |
|-----------|---------------|--------------|
| w/o Visual Features | -3.8% | p<0.001 |
| w/o Bidirectional Attention | -3.4% | p<0.001 |
| w/o Tissue Graph | -4.8% | p<0.001 |
| w/o Cell Graph | -5.9% | p<0.001 |
| Fixed Grid Pooling | -4.3% | p<0.001 |

### Figure 7: Explainability Visualization

<div align="center">
<img src="figures/fig7_explainability.png" width="100%">
</div>

**Multi-Reader Pathologist Study (n=5, 150 patches):**
- **Diagnostic Relevance:** 4.1/5.0 (vs. 2.8 for HACT-Net, p<0.001)
- **Scale Integration:** 4.3/5.0
- **Clinical Trust:** 4.0/5.0

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended 16GB+ VRAM)
- 50GB+ disk space for datasets

### Installation

```bash
# Clone repository
git clone https://github.com/ImamDad/HiGATE.git
cd HiGATE

# Create conda environment (recommended)
conda create -n higate python=3.9
conda activate higate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Data Preparation

Download and organize datasets as follows:

```
data/
├── pannuke/
│   ├── images_fold_1.npy
│   ├── masks_fold_1.npy
│   ├── labels_fold_1.json
│   ├── images_fold_2.npy
│   ├── masks_fold_2.npy
│   ├── labels_fold_2.json
│   ├── images_fold_3.npy
│   ├── masks_fold_3.npy
│   └── labels_fold_3.json
├── monuseg/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
├── digestpath/
│   ├── train.csv
│   ├── test.csv
│   ├── train/
│   │   └── images/
│   └── test/
│       └── images/
└── tcga-brca/
    ├── slides/
    ├── labels.csv
    └── clinical_data.csv
```

### Training

```bash
# Basic training
python scripts/train.py \
    --data_dir /path/to/pannuke \
    --fold fold_1 \
    --checkpoint_dir checkpoints

# Training with W&B logging
python scripts/train.py \
    --data_dir /path/to/pannuke \
    --fold fold_1 \
    --checkpoint_dir checkpoints \
    --use_wandb \
    --wandb_project higate_experiments

# Resume from checkpoint
python scripts/train.py \
    --data_dir /path/to/pannuke \
    --fold fold_1 \
    --checkpoint_dir checkpoints \
    --resume checkpoints/checkpoint_epoch_50.pth
```

### Evaluation

```bash
# Evaluate on PanNuke test set
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --data_dir /path/to/pannuke \
    --fold fold_1 \
    --output_file results/pannuke_results.json
```

### External Validation

```bash
# MoNuSeg segmentation
python scripts/external_validation.py \
    --checkpoint checkpoints/best_model.pth \
    --dataset monuseg \
    --data_dir /path/to/monuseg \
    --output_file results/monuseg_results.json

# DigestPath classification
python scripts/external_validation.py \
    --checkpoint checkpoints/best_model.pth \
    --dataset digestpath \
    --data_dir /path/to/digestpath \
    --output_file results/digestpath_results.json

# TCGA-BRCA WSI grading
python scripts/external_validation.py \
    --checkpoint checkpoints/best_model.pth \
    --dataset tcga \
    --data_dir /path/to/tcga-brca \
    --output_file results/tcga_results.json
```

### Reproduce Paper Figures

```bash
# Generate all figures with synthetic data
python scripts/reproduce_figures.py --save_dir figures

# Generate figures with actual evaluation data
python scripts/reproduce_figures.py \
    --data_file results/evaluation_results.pkl \
    --save_dir figures
```

## 📖 Documentation

### Configuration

HiGATE uses dataclass-based configuration for reproducibility:

```python
from config.default_config import HiGATEConfig

config = HiGATEConfig()
config.model.num_layers = 3
config.model.num_heads = 4
config.training.learning_rate = 1e-4
```

Or use YAML config files:

```yaml
# config.yaml
model:
  num_layers: 3
  num_heads: 4
  dropout: 0.2
training:
  learning_rate: 1e-4
  batch_size: 16
data:
  pannuke_root: /data/pannuke
```

### Custom Dataset Integration

To use HiGATE with your own dataset, implement a dataset class inheriting from `BaseHistoDataset`:

```python
from data.datasets.base_dataset import BaseHistoDataset

class MyDataset(BaseHistoDataset):
    def __getitem__(self, idx):
        # Return dict with keys:
        # 'images': ROI tensor (C, H, W)
        # 'morph': morphological features (6,)
        # 'stardist': StarDist features (12,)
        # 'positions': centroid coordinates (2,)
        # 'labels': class label
        pass
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black .
isort .

# Lint
flake8
```

## 📝 Citation

If you use HiGATE in your research, please cite our paper:

```bibtex
@article{dad2025higate,
  title={HiGATE: Hierarchical Graph Attention for Multi-Scale Tissue Encoder in Computational Pathology},
  author={Dad, Imam and He, Jianfeng and Shen, Tao},
  journal={Journal of Translational Medicine},
  volume={xx},
  number={x},
  pages={xxx--xxx},
  year={2025},
  publisher={BioMed Central},
  doi={xx.xxxx/xxxxx}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

| Author | Affiliation | Role | Contact |
|--------|-------------|------|---------|
| **Imam Dad** | Kunming University of Science and Technology | Lead Author, Software | [![Email](https://img.shields.io/badge/Email-imamdad.csit%40um.uob.edu.pk-red)](mailto:imamdad.csit@um.uob.edu.pk) [![GitHub](https://img.shields.io/badge/GitHub-ImamDad-black)](https://github.com/ImamDad) |
| **Jianfeng He** | Kunming University of Science and Technology | Corresponding Author | [![Email](https://img.shields.io/badge/Email-jfenghe%40kust.edu.cn-blue)](mailto:jfenghe@kust.edu.cn) |
| **Tao Shen** | Third Affiliated Hospital of Kunming Medical University | Clinical Validation | [![Email](https://img.shields.io/badge/Email-shentao%40kmmu.edu.cn-green)](mailto:shentao@kmmu.edu.cn) |

## 🙏 Acknowledgements

We thank the creators of the following datasets for making their data publicly available:
- **PanNuke** - Warwick University
- **MoNuSeg** - Grand Challenge
- **DigestPath** - Grand Challenge
- **TCGA-BRCA** - The Cancer Genome Atlas

This work was supported by:
- **National Natural Science Foundation in China** (No. 82160347)
- **Yunnan Fundamental Research Project** (No. 202301AY070001-251)
- **Yunnan Province Young and Middle-Aged Academic and Technical Leaders Project** (202305AC3500007)

Special thanks to the five board-certified pathologists who participated in the multi-reader study.

## 📧 Contact

For questions, issues, or collaborations:
- **Open an issue** on [GitHub](https://github.com/ImamDad/HiGATE/issues)
- **Email**: imamdad.csit@um.uob.edu.pk
- **Twitter/X**: [@ImamDad](https://twitter.com/ImamDad)

## 🔗 Links

- [📄 Paper](https://doi.org/xxxx)
- [📊 W&B Dashboard](https://wandb.ai/...)
- [🐍 PyPI Package](https://pypi.org/project/higate/)
- [📚 Documentation](https://higate.readthedocs.io)

---

<div align="center">
<b>If you find HiGATE useful for your research, please consider starring ⭐ the repository and citing our paper!</b>
</div>
```

This rewritten README includes:

1. **Professional badges and header** with visual appeal
2. **Clear abstract and key innovations** in table format
3. **State-of-the-art results** summary table
4. **Detailed architecture diagram** with all equations referenced
5. **Comprehensive component explanations** with equations
6. **All figure references** with placeholders for actual images
7. **Detailed results** from the paper including all metrics
8. **Quick start guide** with installation and training commands
9. **Comprehensive documentation** section
10. **Citation information** in BibTeX format
11. **Author information** with contact details
12. **Acknowledgements** for funding and dataset creators
13. **Professional formatting** with emojis and visual elements

The README now provides a complete, professional overview of your research that will help users understand, use, and cite your work effectively.
