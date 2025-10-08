import torch
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

from config import config
from data_processing.feature_extraction import MultiModalFeatureExtractor
from data_processing.graph_construction import HierarchicalGraphBuilder
from data_processing.dataset import PanNukeDataset

logger = logging.getLogger(__name__)

class GraphGenerator:
    """Graph generation pipeline for HiGATE"""
    
    def __init__(self):
        self.feature_extractor = MultiModalFeatureExtractor(config)
        self.graph_builder = HierarchicalGraphBuilder(config)
        self.device = config.DEVICE
        
    def initialize(self) -> bool:
        """Initialize feature extractor"""
        return self.feature_extractor.initialize(self.device)
    
    def generate_graphs_for_dataset(self, dataset: PanNukeDataset, output_dir: Path) -> dict:
        """Generate and save graphs for all samples in dataset"""
        if not self.initialize():
            raise RuntimeError("Failed to initialize graph generator")
        
        # Create output directories
        cell_graph_dir = output_dir / "cell_graphs"
        tissue_graph_dir = output_dir / "tissue_graphs"
        cell_graph_dir.mkdir(parents=True, exist_ok=True)
        tissue_graph_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {
            "total_samples": len(dataset),
            "processed_samples": 0,
            "failed_samples": 0,
            "avg_cell_nodes": 0,
            "avg_tissue_nodes": 0,
            "avg_cell_edges": 0,
            "avg_tissue_edges": 0
        }
        
        total_cell_nodes = 0
        total_tissue_nodes = 0
        total_cell_edges = 0
        total_tissue_edges = 0
        
        for idx in tqdm(range(len(dataset)), desc="Generating graphs"):
            try:
                sample = dataset[idx]
                image = sample['image']
                mask = sample['mask']
                image_id = sample['image_id']
                label = sample['label']
                
                # Extract features
                features = self.feature_extractor.extract_features(image, mask)
                
                if features['num_regions'] == 0:
                    stats['failed_samples'] += 1
                    continue
                
                # Build graphs
                cell_graph = self.graph_builder.build_cell_graph(features)
                tissue_graph, cluster_labels = self.graph_builder.build_tissue_graph(features)
                
                # Add sample information
                cell_graph.image_id = image_id
                cell_graph.y = torch.tensor(label, dtype=torch.float32)
                
                tissue_graph.image_id = image_id  
                tissue_graph.y = torch.tensor(label, dtype=torch.float32)
                tissue_graph.cluster_labels = torch.tensor(cluster_labels, dtype=torch.long)
                
                # Save graphs
                torch.save(cell_graph, cell_graph_dir / f"{image_id}.pt")
                torch.save(tissue_graph, tissue_graph_dir / f"{image_id}.pt")
                
                # Update stats
                total_cell_nodes += cell_graph.num_nodes
                total_cell_edges += cell_graph.num_edges
                total_tissue_nodes += tissue_graph.num_nodes
                total_tissue_edges += tissue_graph.num_edges
                stats['processed_samples'] += 1
                
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {str(e)}")
                stats['failed_samples'] += 1
                continue
        
        # Compute averages
        if stats['processed_samples'] > 0:
            stats['avg_cell_nodes'] = total_cell_nodes / stats['processed_samples']
            stats['avg_tissue_nodes'] = total_tissue_nodes / stats['processed_samples'] 
            stats['avg_cell_edges'] = total_cell_edges / stats['processed_samples']
            stats['avg_tissue_edges'] = total_tissue_edges / stats['processed_samples']
        
        # Save stats
        stats_file = output_dir / "generation_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Graph generation completed: {stats}")
        return stats

def main():
    """Main function for graph generation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate graphs for HiGATE')
    parser.add_argument('--fold', type=str, required=True, choices=['fold1', 'fold2', 'fold3'],
                       help='Fold to generate graphs for')
    args = parser.parse_args()
    
    # Generate graphs for specified fold
    fold_path = config.DATA_ROOT / args.fold
    output_dir = config.GRAPH_DATA_PATH / args.fold
    
    dataset = PanNukeDataset(fold_path)
    generator = GraphGenerator()
    
    stats = generator.generate_graphs_for_dataset(dataset, output_dir)
    print(f"Graph generation stats for {args.fold}: {stats}")

if __name__ == "__main__":
    main()
