import numpy as np
from pathlib import Path
import logging
from config import config
from data_processing.dataset import PanNukeDataset
from data_processing.feature_extraction import MultiModalFeatureExtractor

logger = logging.getLogger(__name__)

def calculate_morphological_statistics():
    """
    Calculate morphological feature statistics from training data
    Used for feature normalization in HiGATE
    """
    print("Calculating morphological statistics...")
    
    # Use training fold for statistics
    dataset = PanNukeDataset(config.TRAIN_FOLD)
    feature_extractor = MultiModalFeatureExtractor(config)
    feature_extractor.initialize(config.DEVICE)
    
    all_morph_features = []
    sample_count = 0
    max_samples = 1000  # Use subset for efficiency
    
    for i in range(min(max_samples, len(dataset))):
        try:
            sample = dataset[i]
            features = feature_extractor.extract_features(sample['image'], sample['mask'])
            
            if features['morph_features'].shape[0] > 0:
                all_morph_features.append(features['morph_features'].numpy())
                sample_count += 1
                
        except Exception as e:
            logger.warning(f"Error processing sample {i}: {str(e)}")
            continue
    
    if all_morph_features:
        all_morph_features = np.concatenate(all_morph_features, axis=0)
        means = np.mean(all_morph_features, axis=0).tolist()
        stds = np.std(all_morph_features, axis=0).tolist()
        
        print(f"Calculated statistics from {sample_count} samples with {all_morph_features.shape[0]} regions")
        print("MORPH_MEAN =", [round(x, 4) for x in means])
        print("MORPH_STD =", [round(x, 4) for x in stds])
        
        # Save to config
        config.MORPH_MEAN = means
        config.MORPH_STD = stds
        
        return means, stds
    else:
        print("No morphological features found!")
        return None, None

if __name__ == "__main__":
    calculate_morphological_statistics()
