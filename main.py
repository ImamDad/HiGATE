
import argparse
import os
import logging
from pathlib import Path
import sys
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config import config
from training.trainer import HiGATETrainer
from models.higate import HiGATE
from data_processing.dataset import PanNukeDataset
from training.explainability import HierarchicalGNNExplainer
from training.external_validation import ExternalValidator
from training.evaluate import Evaluator

def setup_logging():
    """Configure logging with both console and file output"""
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.LOG_DIR / 'higate.log')
        ]
    )
    return logging.getLogger(__name__)

def setup_environment():
    """Configure system environment for the project"""
    # Optimize CUDA performance
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    # Set environment variables
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="HiGATE: Hierarchical Graph Attention for Computational Pathology",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--mode', type=str, required=True,
                      choices=['train', 'evaluate', 'explain', 'validate', 'generate_graphs'],
                      help="Operation mode")
    parser.add_argument('--model-path', type=str, 
                      default=str(config.MODEL_SAVE_PATH / 'best_model.pth'),
                      help="Path to model checkpoint")
    parser.add_argument('--fold', type=str, default='fold1',
                      choices=['fold1', 'fold2', 'fold3'],
                      help="Fold to use for training/validation")
    parser.add_argument('--num-samples', type=int, default=5,
                      help="Number of samples for explanation")
    parser.add_argument('--debug', action='store_true',
                      help="Enable debug mode")
    
    return parser.parse_args()

def validate_paths():
    """Ensure all required directories exist"""
    required_dirs = [
        config.LOG_DIR,
        config.MODEL_SAVE_PATH,
        config.RESULTS_PATH,
        config.GRAPH_DATA_PATH
    ]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)

def main():
    """Main execution function"""
    # Initial setup
    setup_environment()
    logger = setup_logging()
    args = parse_arguments()
    validate_paths()

    try:
        logger.info(f"Starting HiGATE in {args.mode} mode")
        
        if args.mode == 'train':
            from scripts.generate_graphs import GraphGenerator
            from data_processing.dataset import PanNukeDataset
            
            # Generate graphs if not exists
            generator = GraphGenerator()
            for fold_name in ['fold1', 'fold2', 'fold3']:
                dataset = PanNukeDataset(config.DATA_ROOT / fold_name)
                output_dir = config.GRAPH_DATA_PATH / fold_name
                generator.generate_graphs_for_dataset(dataset, output_dir)
            
            # Initialize and train model
            model = HiGATE(config)
            trainer = HiGATETrainer(model, config)
            trainer.train()
            
        elif args.mode == 'evaluate':
            evaluator = Evaluator(args.model_path)
            results = evaluator.evaluate()
            logger.info(f"Evaluation results: {results}")
            
        elif args.mode == 'explain':
            explainer = HierarchicalGNNExplainer(args.model_path)
            from training.explainability import explain_model_on_dataset
            explain_model_on_dataset(args.model_path, args.num_samples)
            
        elif args.mode == 'validate':
            validator = ExternalValidator(args.model_path)
            results = validator.run_all_validations(Path("data/external"))
            logger.info(f"External validation results: {results}")
            
        elif args.mode == 'generate_graphs':
            from scripts.generate_graphs import GraphGenerator
            from data_processing.dataset import PanNukeDataset
            
            generator = GraphGenerator()
            for fold_name in ['fold1', 'fold2', 'fold3']:
                dataset = PanNukeDataset(config.DATA_ROOT / fold_name)
                output_dir = config.GRAPH_DATA_PATH / fold_name
                stats = generator.generate_graphs_for_dataset(dataset, output_dir)
                logger.info(f"Generated graphs for {fold_name}: {stats}")

    except Exception as e:
        logger.error(f"Error in {args.mode} mode: {str(e)}", exc_info=True)
        sys.exit(1)
        
    logger.info("HiGATE execution completed successfully")

if __name__ == "__main__":
    main()
