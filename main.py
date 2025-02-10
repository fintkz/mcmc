import numpy as np
import time
import pandas as pd
import json
from itertools import combinations
from pathlib import Path
import torch
from data import SyntheticDataGenerator, DatasetFeatures
from models.prophet_model import ProphetModel
from models.temporal_fusion import TFTModel
from models.bayesian_ensemble import GPUBayesianEnsemble
from utils import evaluate_predictions
import os
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import argparse
from concurrent.futures import ProcessPoolExecutor
import torch.multiprocessing as mp
from functools import partial


def setup_logging():
    """Setup logging configuration"""
    log_dir = Path("results")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"work_log_{timestamp}.txt"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def prepare_feature_combinations(data: DatasetFeatures, logger: logging.Logger):
    """Generate all possible feature combinations with named tensors"""
    # Remove names before getting sizes
    features = data.features.rename(None)
    temporal = data.temporal.rename(None)

    # Get feature names using integer indexing
    feature_names = [f"feature_{i}" for i in range(features.size(1))]  # dim 1 is features
    temporal_names = [f"temporal_{i}" for i in range(temporal.size(1))]  # dim 1 is temporal_features

    # Generate all possible combinations of feature indices
    all_combinations = []
    for r in range(1, len(feature_names) + 1):
        all_combinations.extend(combinations(range(len(feature_names)), r))

    # Add baseline (all features)
    all_combinations = [tuple()] + list(all_combinations)

    logger.info(f"Generated {len(all_combinations)} feature combinations")
    return all_combinations


def process_single_model(combo: tuple, data: DatasetFeatures, logger: logging.Logger, model_name: str) -> tuple:
    """Process a single combination for a specific model"""
    combo_name = "_".join(map(str, combo)) if combo else "baseline"
    logger.info(f"\nTraining {model_name} for combination: {combo_name}")
    
    try:
        device = torch.device("cuda:0")  # Use first GPU
        logger.debug(f"Using device: {device}")
        
        # Move data to device first
        features = data.features.to(device)
        temporal = data.temporal.to(device)
        target = data.target.to(device)
        logger.debug(f"Data moved to {device}")
        
        # Restore tensor names after moving to device and multiprocessing
        features = features.refine_names('time', 'features')
        temporal = temporal.refine_names('time', 'temporal_features')
        target = target.refine_names('time')
        logger.debug("Tensor names restored")
        
        # Select features based on combination
        if combo:
            # Remove names for selection, then restore
            features = features.rename(None)
            features = features.index_select(
                1,  # features dimension is 1 (time=0, features=1)
                torch.tensor(list(combo), device=device)  # Create tensor on same device
            )
            features = features.refine_names('time', 'features')
            logger.debug(f"Selected features: {combo}")
            
            # Log feature statistics
            features_np = features.rename(None).cpu().numpy()
            for i, feat_idx in enumerate(combo):
                feat_stats = {
                    'mean': float(np.mean(features_np[:, i])),
                    'std': float(np.std(features_np[:, i])),
                    'min': float(np.min(features_np[:, i])),
                    'max': float(np.max(features_np[:, i]))
                }
                logger.info(f"Feature {feat_idx} stats: {feat_stats}")
                
                # Check correlation with target
                target_np = target.rename(None).cpu().numpy()
                corr = np.corrcoef(features_np[:, i], target_np)[0, 1]
                logger.info(f"Feature {feat_idx} correlation with target: {corr:.3f}")
        
        # Prepare data based on model type
        if model_name == "prophet":
            if combo:
                # Move to CPU for Prophet (it doesn't use GPU)
                features_for_model = features.rename(None).cpu().numpy()
                feature_names = [str(i) for i in combo]
                logger.info(f"Training Prophet model with features: {feature_names}")
                
                # Log feature importance (correlation based)
                target_np = target.rename(None).cpu().numpy()
                correlations = [np.corrcoef(features_for_model[:, i], target_np)[0, 1] for i in range(features_for_model.shape[1])]
                for feat_name, corr in zip(feature_names, correlations):
                    logger.info(f"Feature {feat_name} correlation with target: {corr:.3f}")
                
                model = ProphetModel()
                preds = model.train_and_predict(
                    data.dates,
                    target.rename(None).cpu().numpy(),
                    features=features_for_model,
                    feature_names=feature_names
                )
            else:
                logger.info("Training baseline Prophet model (no features)")
                model = ProphetModel()
                preds = model.train_and_predict(
                    data.dates,
                    target.rename(None).cpu().numpy()
                )
            
            # Evaluate predictions
            metrics = evaluate_predictions(
                target.rename(None).cpu().numpy(),
                preds
            )
            logger.info(f"Combination {combo_name} - RMSE: {metrics['rmse']:.2f}, MAPE: {metrics['mape']:.2f}%")
            
            result = {
                "predictions": preds.tolist(),
                "metrics": metrics
            }
            
        return combo_name, result
        
    except Exception as e:
        logger.error(f"Model training failed for {combo_name}: {str(e)}")
        return combo_name, None


def train_parallel(
    data: DatasetFeatures,
    logger: logging.Logger,
    model_name: str = None,
    num_workers: int = 4,
    results_path: str = "results/model_results.json"
) -> None:
    """Train models in parallel using process pool"""
    # Load existing results
    results = load_existing_results(results_path)
    
    # Update feature_dates and actual if they're empty
    if not results["feature_dates"]:
        results["feature_dates"] = data.feature_dates
    if not results["actual"]:
        results["actual"] = data.target.rename(None).cpu().tolist()

    # Get all feature combinations
    all_combinations = prepare_feature_combinations(data, logger)
    logger.info(f"Starting parallel training with {num_workers} workers")
    
    # Remove tensor names for multiprocessing
    data_unnamed = DatasetFeatures(
        features=data.features.rename(None),
        temporal=data.temporal.rename(None),
        target=data.target.rename(None),
        dates=data.dates,
        feature_dates=data.feature_dates
    )
    
    # Create partial function with fixed arguments
    process_func = partial(process_single_model, data=data_unnamed, logger=logger, model_name=model_name)
    
    # Process combinations in parallel
    completed = 0
    total = len(all_combinations)
    successful = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for combo_name, result in executor.map(process_func, all_combinations):
            if result is not None:
                # Update results
                if combo_name not in results["predictions"]:
                    results["predictions"][combo_name] = {}
                results["predictions"][combo_name][model_name] = result
                successful += 1
            else:
                failed += 1
                
            # Save periodically
            completed += 1
            if completed % 5 == 0:
                save_results(results, results_path)
                logger.info(f"Completed {completed}/{total} combinations (Success: {successful}, Failed: {failed})")
    
    # Final save
    save_results(results, results_path)
    logger.info(f"Training complete. Total combinations: {total}, Successful: {successful}, Failed: {failed}")


def load_existing_results(results_path: str) -> dict:
    """Load existing results from JSON file, create if doesn't exist"""
    if Path(results_path).exists():
        with open(results_path, 'r') as f:
            return json.load(f)
    return {
        "feature_dates": {},
        "actual": [],
        "predictions": {}
    }


def save_results(results: dict, results_path: str):
    """Save results to JSON file"""
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate forecasting models')
    parser.add_argument('--model', type=str, choices=['prophet', 'tft', 'bayesian', 'all'],
                      help='Which model to train (prophet, tft, bayesian, or all)')
    parser.add_argument('--results-path', type=str, default='results/model_results.json',
                      help='Path to save/load results JSON')
    parser.add_argument('--workers', type=int, default=4,
                      help='Number of worker threads for parallel training')
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()

    try:
        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)

        # Set multiprocessing start method to 'spawn' for CUDA support
        mp.set_start_method('spawn')
        
        # Log system information
        if torch.cuda.is_available():
            logger.info("CUDA available: True")
            logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            logger.warning("CUDA is not available, using CPU")

        # Create results directory
        Path("results").mkdir(exist_ok=True)

        # Generate synthetic data
        logger.info("Generating synthetic data...")
        generator = SyntheticDataGenerator()
        data = generator.generate_data()

        # Train specific model
        train_parallel(data, logger, args.model, args.workers, args.results_path)

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
