import torch.nn as nn
from config import config

class CellTypeClassifier(nn.Module):
    """Final classifier for cell type prediction (Section 3.5)"""
    
    def __init__(self, input_dim: int = 512, num_classes: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        return self.mlp(x)
