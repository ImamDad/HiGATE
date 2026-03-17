
# HiGATE: Hierarchical Graph Attention for Multi-Scale Tissue Encoder in Computational Pathology

[![Paper](https://img.shields.io/badge/Paper-Journal%20of%20Translational%20Medicine-blue)](https://doi.org/xxxx)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/ImamDad/HiGATE)](https://github.com/ImamDad/HiGATE)

Official PyTorch implementation of HiGATE, a state-of-the-art framework for multi-scale tissue analysis in computational pathology.


## Overview

Histopathological diagnosis is inherently a multi-scale reasoning process where pathologists seamlessly integrate cellular morphology with tissue architecture. HiGATE (Hierarchical Graph Attention Tissue Encoder) introduces a biologically-inspired framework that unifies cellular and tissue-level analysis through a novel dual-graph architecture with bidirectional cross-level attention.

Key Innovations:
- Bidirectional Cross-Level Attention: Dynamic information exchange between cellular and tissue scales (+3.4% accuracy improvement)
- Learnable Spatially-Constrained Graph Construction: Adaptive region formation via differentiable pooling (+4.5% over fixed methods)
- Multi-Modal Feature Fusion: Integration of DINOv2, morphological, and StarDist features (+2.2% over concatenation)
- Inherent Explainability: Multi-scale explanations validated by pathologists (4.1/5.0 rating)



## State-of-the-Art Results

### PanNuke Nuclei Classification

| Model | Accuracy | F1-Score | AUROC | AUPRC |
|-------|----------|----------|-------|-------|
| HiGATE (Ours) | 91.3% | 0.896 | 0.958 | 0.901 |
| HACT-Net | 89.1% | 0.879 | 0.945 | 0.872 |
| Swin Transformer | 87.4% | 0.858 | 0.933 | 0.841 |
| TransPath | 87.8% | 0.861 | 0.938 | 0.849 |

### Cross-Dataset Validation

| Dataset | Task | Metric | HiGATE | HACT-Net | Improvement |
|---------|------|--------|--------|----------|-------------|
| MoNuSeg | Segmentation | Dice | 0.841 | 0.796 | +4.5% |
| DigestPath | Classification | Accuracy | 87.2% | 83.2% | +4.0% |
| TCGA-BRCA | WSI Grading | Accuracy | 85.4% | 83.5% | +1.9% |



## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (8GB+ VRAM recommended)
- 50GB+ disk space for datasets

### From Source

git clone https://github.com/ImamDad/HiGATE.git
cd HiGATE
conda create -n higate python=3.9
conda activate higate
pip install -r requirements.txt
pip install -e .

### Via pip

pip install higate


### Verify Installation

pytest tests/
python -c "from models.higate import HiGATE; model = HiGATE()"




## Data Preparation

### Download Datasets

| Dataset | Task | Download Link |
|---------|------|---------------|
| PanNuke | Nuclei Classification | [Warwick University](https://warwick.ac.uk/fac/sci/dcs/research/tia/data/pannuke) |
| MoNuSeg | Segmentation | [Grand Challenge](https://monuseg.grand-challenge.org) |
| DigestPath | Classification | [Grand Challenge](https://digestpath2019.grand-challenge.org) |
| TCGA-BRCA | WSI Grading | [GDC Portal](https://portal.gdc.cancer.gov/projects/TCGA-BRCA) |

### Directory Structure

data/
├── pannuke/
│   ├── fold0/
│   │   ├── extracted_images_npy/
│   │   ├── extracted_masks/
│   │   ├── extracted_cell_counts.csv
│   │   └── extracted_types.csv
│   ├── fold1/
│   └── fold2/
├── monuseg/
│   ├── train/
│   └── test/
├── digestpath/
│   ├── train.csv
│   ├── test.csv
│   └── images/
└── tcga-brca/
    ├── slides/
    └── labels.csv



## Usage

### Training

# Basic training
python scripts/train.py --data_dir /path/to/pannuke --fold fold0

# With W&B logging
python scripts/train.py --data_dir /path/to/pannuke --fold fold0 --use_wandb

# Resume from checkpoint
python scripts/train.py --data_dir /path/to/pannuke --fold fold0 --resume checkpoints/checkpoint.pth


### Evaluation

python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --data_dir /path/to/pannuke \
    --fold fold0 \
    --output_file results.json


### External Validation

# MoNuSeg
python scripts/external_validation.py \
    --checkpoint checkpoints/best_model.pth \
    --dataset monuseg \
    --data_dir /path/to/monuseg

# DigestPath
python scripts/external_validation.py \
    --checkpoint checkpoints/best_model.pth \
    --dataset digestpath \
    --data_dir /path/to/digestpath

# TCGA-BRCA
python scripts/external_validation.py \
    --checkpoint checkpoints/best_model.pth \
    --dataset tcga \
    --data_dir /path/to/tcga-brca


### Reproduce Figures

python scripts/reproduce_figures.py --save_dir figures




## Architecture


Input WSI/Patch
    ↓
Multi-Modal Feature Extraction
├── DINOv2 (visual)
├── Morphological (6 features)
└── StarDist (12 features)
    ↓
Attention-Weighted Fusion
    ↓
Cell Graph Construction (learnable adjacency, k=20)
    ↓
Differentiable Pooling with Spatial Regularization
    ↓
┌─────────────────┐    ┌─────────────────┐
│ Cell GAT Layers │    │ Tissue GAT Layers│
│ (3 layers)      │    │ (3 layers)      │
└────────┬────────┘    └────────┬────────┘
         └──────────┬───────────┘
                    ↓
    Bidirectional Cross-Level Attention
    ├── Bottom-Up: cells → tissue
    └── Top-Down: tissue → cells
                    ↓
┌─────────────────┐    ┌─────────────────┐
│ Cell Attention  │    │ Tissue Attention│
│ Pooling         │    │ Pooling         │
└────────┬────────┘    └────────┬────────┘
         └──────────┬───────────┘
                    ↓
         Classification Head


## Key Components

### Feature Extractors
| Module | Input | Output | Description |
|--------|-------|--------|-------------|
| `DINOv2FeatureExtractor` | (B,3,224,224) | (B,256) | Visual features |
| `MorphologicalFeatureExtractor` | (B,6) | (B,128) | Geometric descriptors |
| `StarDistFeatureExtractor` | (B,12) | (B,128) | Nuclear morphometrics |
| `AttentionFusion` | List | (B,512) | Multi-modal fusion |

### Graph Construction
| Module | Input | Output | Description |
|--------|-------|--------|-------------|
| `LearnableAdjacency` | (N,512), (N,2) | (N,N) | Learnable graph edges |
| `DifferentiablePooling` | (N,512), edge_index | (N,K), L_spatial | Soft region assignment |
| `HierarchicalGraphBuilder` | (N,512), (N,2) | cell_graph, tissue_graph | Complete graph construction |

### Attention Mechanisms
| Module | Input | Output | Description |
|--------|-------|--------|-------------|
| `GATLayer` | (N,d), edge_index | (N,d) | Graph attention |
| `BidirectionalCrossLevelAttention` | h_cell, h_tissue, S | h_cell, h_tissue | Cross-scale attention |



## Code Structure


higate/
├── config/           # Configuration files
├── data/            # Dataset classes
│   └── datasets/    # PanNuke, MoNuSeg, DigestPath, TCGA-BRCA
├── models/          # Model architecture
│   ├── components/  # Modular components
│   └── higate.py    # Main model
├── training/        # Training utilities
├── utils/           # Helper functions
├── scripts/         # Run scripts
├── tests/           # Unit tests
├── requirements.txt # Dependencies
├── setup.py         # Package setup
└── README.md        # This file


## Testing

# Run all tests
pytest tests/

# Run specific tests
pytest tests/test_models.py -v
pytest tests/test_data.py -v

# With coverage
pytest tests/ --cov=higate --cov-report=html


## Citation

If you use HiGATE in your research, please cite:

bibtex
@article{dad2025higate,
  title={HiGATE: Hierarchical Graph Attention for Multi-Scale Tissue Encoder in Computational Pathology},
  author={Dad, Imam and He, Jianfeng and Shen, Tao},
  journal={Journal of Translational Medicine},
  year={2025}
}




## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



## Acknowledgements

This work was supported by:
- National Natural Science Foundation in China (No. 82160347)
- Yunnan Fundamental Research Project (No. 202301AY070001-251)
- Yunnan Province Young and Middle-Aged Academic and Technical Leaders Project (202305AC3500007)

We thank the creators of PanNuke, MoNuSeg, DigestPath, and TCGA-BRCA datasets.



## Contact

Imam Dad - Lead Author, Software  
Email: imamdad.csit@um.uob.edu.pk  
GitHub: [@ImamDad](https://github.com/ImamDad)

Jianfeng He - Corresponding Author  
Email: jfenghe@kust.edu.cn

Tao Shen - Clinical Validation  
Email: shentao@kmmu.edu.cn


[![GitHub stars](https://img.shields.io/github/stars/ImamDad/HiGATE?style=social)](https://github.com/ImamDad/HiGATE)


This README is clean, well-structured, and GitHub-friendly with:
- Clear sections using headers
- Simple markdown formatting that renders properly
- Code blocks with proper syntax highlighting
- Tables for results and component summaries
- Badges for quick information
- Minimal decorative elements that might not render correctly
