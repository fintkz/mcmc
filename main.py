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
import torch.distributed as dist


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
    """Generate all possible feature combinations"""
    # Get feature dimensions
    n_features = data.features.size(1)  # dim 1 is features
    n_temporal = data.temporal.size(1)  # dim 1 is temporal_features

    # Generate all possible combinations of feature indices
    all_combinations = []
    for r in range(1, len(n_features) + 1):
        all_combinations.extend(combinations(range(n_features), r))

    # Add baseline (all features)
    all_combinations = [tuple()] + list(all_combinations)

    logger.info(f"Generated {len(all_combinations)} feature combinations")
    return all_combinations


def safe_corrcoef(x, y):
    """Calculate correlation coefficient safely handling edge cases"""
    # Check for constant values
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0

    # Check for NaN/inf values
    if (
        np.any(np.isnan(x))
        or np.any(np.isnan(y))
        or np.any(np.isinf(x))
        or np.any(np.isinf(y))
    ):
        return 0.0

    return np.corrcoef(x, y)[0, 1]


def process_single_model(
    combo: tuple, data: DatasetFeatures, logger: logging.Logger, model_name: str
) -> tuple:
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

        # Select features based on combination
        if combo:
            features = features.index_select(
                1,  # features dimension is 1 (time=0, features=1)
                torch.tensor(
                    list(combo), device=device
                ),  # Create tensor on same device
            )
            logger.debug(f"Selected features: {combo}")

            # Log feature statistics
            features_np = features.cpu().numpy()
            target_np = target.cpu().numpy()

            logger.info("\nFeature Diagnostics:")
            for i, feat_idx in enumerate(combo):
                feat_data = features_np[:, i]

                # Basic statistics
                feat_stats = {
                    "mean": float(np.mean(feat_data)),
                    "std": float(np.std(feat_data)),
                    "min": float(np.min(feat_data)),
                    "max": float(np.max(feat_data)),
                    "zeros": int(np.sum(feat_data == 0)),
                    "nans": int(np.sum(np.isnan(feat_data))),
                    "infs": int(np.sum(np.isinf(feat_data))),
                }

                # Correlation with target
                corr = safe_corrcoef(feat_data, target_np)

                logger.info(f"\nFeature {feat_idx}:")
                logger.info(f"  Statistics: {feat_stats}")
                logger.info(f"  Correlation with target: {corr:.3f}")

                if feat_stats["std"] == 0:
                    logger.warning(
                        f"  WARNING: Feature {feat_idx} has zero standard deviation (constant value)"
                    )
                if feat_stats["nans"] > 0:
                    logger.warning(
                        f"  WARNING: Feature {feat_idx} has {feat_stats['nans']} NaN values"
                    )
                if feat_stats["infs"] > 0:
                    logger.warning(
                        f"  WARNING: Feature {feat_idx} has {feat_stats['infs']} infinite values"
                    )

        # Prepare data based on model type
        if model_name == "prophet":
            if combo:
                # Convert features to numpy for Prophet
                features_for_model = features.cpu().numpy()
                logger.info(f"\nTraining Prophet model with features: {combo}")

                model = ProphetModel()
                preds = model.train_and_predict(
                    data.dates,
                    target.cpu().numpy(),
                    features=features_for_model,
                )
            else:
                logger.info("\nTraining baseline Prophet model (no features)")
                model = ProphetModel()
                preds = model.train_and_predict(data.dates, target.cpu().numpy())

        elif model_name == "bayesian":
            # Initialize model
            input_dim = features.size(1) if combo else 1
            model = GPUBayesianEnsemble(input_dim=input_dim, device=str(device))

            if combo:
                logger.info(f"\nTraining Bayesian model with features: {combo}")
                try:
                    # Train the model
                    model.train(features, target)
                    predictions, uncertainties = model.predict(features)
                    preds = predictions  # We only need the mean for evaluation
                except Exception as e:
                    logger.error(f"Error training Bayesian model: {str(e)}")
                    raise
            else:
                # For baseline, use time index as feature
                time_feature = (
                    torch.arange(len(target), device=device).float().reshape(-1, 1)
                )
                logger.info("\nTraining baseline Bayesian model (time only)")
                try:
                    # Train the model
                    model.train(time_feature, target)
                    predictions, uncertainties = model.predict(time_feature)
                    preds = predictions  # We only need the mean for evaluation
                except Exception as e:
                    logger.error(f"Error training baseline Bayesian model: {str(e)}")
                    raise

            # Move predictions to CPU for evaluation
            preds = preds.cpu().numpy()

        elif model_name == "tft":
            # Initialize model
            input_dim = features.size(1) if combo else 1
            model = TFTModel(num_features=input_dim, device=str(device))

            if combo:
                logger.info(f"\nTraining TFT model with features: {combo}")
                try:
                    # Train the model
                    model.train(features, target)
                    preds = model.predict(features)
                except Exception as e:
                    logger.error(f"Error training TFT model: {str(e)}")
                    raise
            else:
                # For baseline, use time index as feature
                time_feature = (
                    torch.arange(len(target), device=device).float().reshape(-1, 1)
                )
                logger.info("\nTraining baseline TFT model (time only)")
                try:
                    # Train the model
                    model.train(time_feature, target)
                    preds = model.predict(time_feature)
                except Exception as e:
                    logger.error(f"Error training baseline TFT model: {str(e)}")
                    raise

            # Move predictions to CPU for evaluation if they're not already
            if isinstance(preds, torch.Tensor):
                preds = preds.cpu().numpy()

        else:
            raise ValueError(f"Unknown model type: {model_name}")

        # Evaluate predictions
        metrics = evaluate_predictions(target.cpu().numpy(), preds)
        logger.info(f"\nResults for {combo_name}:")
        logger.info(f"  RMSE: {metrics['rmse']:.2f}")
        logger.info(f"  MAPE: {metrics['mape']:.2f}%")

        result = {"predictions": preds.tolist(), "metrics": metrics}

        return combo_name, result

    except Exception as e:
        logger.error(f"Model training failed for {combo_name}: {str(e)}")
        return combo_name, None


def train_parallel(
    data: DatasetFeatures,
    logger: logging.Logger,
    model_name: str = None,
    num_workers: int = 4,
    results_path: str = "results/model_results.json",
) -> None:
    """Train models in parallel using process pool"""
    # Load existing results
    results = load_existing_results(results_path)

    # Update feature_dates and actual if they're empty
    if not results["feature_dates"]:
        results["feature_dates"] = data.feature_dates
    if not results["actual"]:
        results["actual"] = data.target.cpu().tolist()

    # Get all feature combinations
    all_combinations = prepare_feature_combinations(data, logger)
    logger.info(f"Starting parallel training with {num_workers} workers")

    # Create partial function with fixed arguments
    process_func = partial(
        process_single_model, data=data, logger=logger, model_name=model_name
    )

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
                logger.info(
                    f"Completed {completed}/{total} combinations (Success: {successful}, Failed: {failed})"
                )

    # Final save
    save_results(results, results_path)
    logger.info(
        f"Training complete. Total combinations: {total}, Successful: {successful}, Failed: {failed}"
    )


def load_existing_results(results_path: str) -> dict:
    """Load existing results from JSON file, create if doesn't exist"""
    if Path(results_path).exists():
        with open(results_path, "r") as f:
            return json.load(f)
    return {"feature_dates": {}, "actual": [], "predictions": {}}


def save_results(results: dict, results_path: str):
    """Save results to JSON file"""
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)


def setup(rank, world_size):
    """Initialize distributed training"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """Cleanup distributed training"""
    dist.destroy_process_group()


def train_model(rank, world_size, model_type="tft"):
    """Train model on a single GPU

    Args:
        rank: GPU rank
        world_size: Total number of GPUs
        model_type: Type of model to train (tft, bayesian, or prophet)
    """
    # Initialize distributed training
    setup(rank, world_size)

    try:
        # Generate data
        dataset = SyntheticDataGenerator().generate_data()

        # Move data to device
        X = dataset.features.to(f"cuda:{rank}")
        y = dataset.target.to(f"cuda:{rank}")

        # Create model based on type
        if model_type == "tft":
            model = TFTModel(num_features=X.size(1), device=f"cuda:{rank}", rank=rank)
        elif model_type == "bayesian":
            model = GPUBayesianEnsemble(
                input_dim=X.size(1), device=f"cuda:{rank}", rank=rank
            )
        elif model_type == "prophet":
            # Prophet doesn't need GPU/distributed training
            if rank == 0:
                model = ProphetModel()
                predictions = model.train_and_predict(
                    dates=dataset.dates,
                    y=y.cpu().numpy(),
                    features=X.cpu().numpy(),
                )
                metrics = evaluate_predictions(y.cpu(), torch.tensor(predictions))
                print(f"Prophet Metrics: {metrics}")
            cleanup()
            return

        # Train model
        model.train(X, y)

        # Generate predictions on rank 0
        if rank == 0:
            model.eval()
            with torch.no_grad():
                predictions = model.predict(X)
                metrics = evaluate_predictions(y.cpu(), predictions.cpu())
                print(f"{model_type.upper()} Metrics: {metrics}")

    except Exception as e:
        print(f"Error on rank {rank}: {str(e)}")
        raise
    finally:
        cleanup()


def main():
    """Main training function"""
    # Setup logging
    logger = setup_logging()

    # Log process information
    logger.info(f"Starting training process with PID: {os.getpid()}")

    try:
        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)

        # CUDA optimizations
        if torch.cuda.is_available():
            # Enable TF32 for better performance on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Set to fastest mode
            torch.backends.cudnn.benchmark = True

            # Log CUDA memory status
            logger.info(
                f"CUDA Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
            )
            logger.info(
                f"CUDA Memory Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB"
            )

        # Set multiprocessing start method to 'spawn' for CUDA support
        mp.set_start_method("spawn")

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

        # Get number of available GPUs
        world_size = torch.cuda.device_count()
        if world_size < 1:
            raise RuntimeError("No CUDA GPUs available")

        print(f"Training on {world_size} GPUs")

        # Train each model type
        for model_type in ["tft", "bayesian", "prophet"]:
            print(f"\nTraining {model_type.upper()} model...")
            mp.spawn(
                train_model,
                args=(world_size, model_type),
                nprocs=world_size,
                join=True,
            )

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
