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
    features = data.features  # [time, features]
    temporal = data.temporal  # [time, temporal_features]
    
    # Get feature names
    feature_names = [f"feature_{i}" for i in range(features.size('features'))]
    temporal_names = [f"temporal_{i}" for i in range(temporal.size('temporal_features'))]
    
    # Generate all possible combinations of feature indices
    all_combinations = []
    for r in range(1, len(feature_names) + 1):
        all_combinations.extend(combinations(range(len(feature_names)), r))
    
    # Add baseline (all features)
    all_combinations = [tuple()] + list(all_combinations)
    
    logger.info(f"Generated {len(all_combinations)} feature combinations")
    return all_combinations


def process_gpu_task(task: tuple) -> tuple:
    """Process a single GPU task with named tensors"""
    combo, gpu_id, data, logger = task
    combo_name = "_".join(map(str, combo)) if combo else "baseline"
    
    try:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)
        logger.info(f"\nTraining models for combination: {combo_name} on GPU {gpu_id}")

        # Select features based on combination
        if combo:
            features = data.features.index_select('features', 
                torch.tensor(list(combo), device=data.features.device))
        else:
            features = data.features

        # Log initial shapes
        logger.info(f"Initial features shape: time={features.size('time')}, features={features.size('features')}")
        logger.info(f"Temporal features shape: time={data.temporal.size('time')}, temporal_features={data.temporal.size('temporal_features')}")

        # Rename temporal features dimension to match features before concatenating
        temporal_aligned = data.temporal.rename(None).rename('time', 'features')

        # Combine features and temporal features (using unnamed tensors for concatenation)
        X = torch.cat([
            features.rename(None),
            temporal_aligned.rename(None)
        ], dim=1).refine_names('time', 'features')
        
        y = data.target

        # Move tensors to device
        X = X.to(device)
        y = y.to(device)

        # Log pre-scaling shapes
        logger.info(f"Combined input shape before scaling: time={X.size('time')}, features={X.size('features')}")
        logger.info(f"Target shape before scaling: time={y.size('time')}")

        # Scale features
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        # Scale while preserving names
        try:
            X_scaled = torch.tensor(
                scaler_X.fit_transform(X.rename(None).cpu().numpy()),
                device=device
            ).refine_names('time', 'features')
        except ValueError as e:
            logger.error(f"Failed to scale features: {str(e)}")
            raise
        
        y_scaled = torch.tensor(
            scaler_y.fit_transform(y.rename(None).cpu().numpy().reshape(-1, 1)),
            device=device
        ).squeeze(-1).refine_names('time')

        # Log post-scaling shapes
        logger.info(f"Input shape after scaling: time={X_scaled.size('time')}, features={X_scaled.size('features')}")
        logger.info(f"Target shape after scaling: time={y_scaled.size('time')}")

        # Initialize models
        prophet_model = ProphetModel()
        
        tft_model = TFTModel(
            num_features=X_scaled.size('features'),
            seq_length=30,
            batch_size=32,
            device=device
        )
        
        bayesian_model = GPUBayesianEnsemble(
            input_dim=X_scaled.size('features'),
            device=device,
            batch_size=32
        )

        # Train Prophet (CPU model)
        logger.info(f"Training Prophet for {combo_name}")
        prophet_preds = prophet_model.train_and_predict(
            data.dates, y.rename(None).cpu().numpy()
        )

        # Train TFT
        logger.info(f"Training TFT for {combo_name}")
        tft_model.train(X_scaled, y_scaled)
        tft_preds = tft_model.predict(X_scaled)
        # Inverse transform predictions
        tft_preds = scaler_y.inverse_transform(
            tft_preds.reshape(-1, 1)
        ).flatten()

        # Train Bayesian
        logger.info(f"Training Bayesian for {combo_name}")
        bayesian_model.train(X_scaled, y_scaled)
        bayesian_mean, bayesian_std = bayesian_model.predict(X_scaled)
        # Inverse transform predictions and uncertainty
        bayesian_mean = scaler_y.inverse_transform(
            bayesian_mean.reshape(-1, 1)
        ).flatten()
        bayesian_std = bayesian_std * scaler_y.scale_[0]

        # Move predictions to CPU for evaluation
        y_cpu = y.rename(None).cpu().numpy()

        # Evaluate predictions
        result = {
            "prophet": {
                "predictions": prophet_preds.tolist(),
                "metrics": evaluate_predictions(y_cpu, prophet_preds)
            },
            "tft": {
                "predictions": tft_preds.tolist(),
                "metrics": evaluate_predictions(y_cpu, tft_preds)
            },
            "bayesian": {
                "predictions": bayesian_mean.tolist(),
                "uncertainty": bayesian_std.tolist(),
                "metrics": evaluate_predictions(y_cpu, bayesian_mean)
            }
        }

        # Log metrics
        logger.info(f"Prophet MAPE: {result['prophet']['metrics']['mape']:.2f}%")
        logger.info(f"TFT MAPE: {result['tft']['metrics']['mape']:.2f}%")
        logger.info(f"Bayesian MAPE: {result['bayesian']['metrics']['mape']:.2f}%")

        # Clean up GPU memory
        del X_scaled, y_scaled, tft_model, bayesian_model
        torch.cuda.empty_cache()

        return combo_name, result

    except Exception as e:
        logger.error(f"Model training failed for {combo_name}: {str(e)}")
        torch.cuda.empty_cache()
        raise

    finally:
        # Ensure GPU memory is cleaned up even if successful
        torch.cuda.empty_cache()

def train_and_evaluate_all_models(data: DatasetFeatures, logger: logging.Logger) -> dict:
    """Train and evaluate all models with named tensors"""
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    num_gpus = torch.cuda.device_count()
    logger.info(f"Training using {num_gpus} GPUs")

    # Generate feature combinations
    combinations = prepare_feature_combinations(data, logger)

    # Prepare tasks
    tasks = []
    for i, combo in enumerate(combinations):
        gpu_id = i % num_gpus
        tasks.append((combo, gpu_id, data, logger))

    # Initialize results
    results = {
        "feature_dates": data.feature_dates,
        "actual": data.target.rename(None).tolist(),
        "predictions": {},
    }

    # Process tasks
    for task in tasks:
        combo_name, result = process_gpu_task(task)
        results["predictions"][combo_name] = result
        torch.cuda.empty_cache()

    return results


def main():
    # Setup logging
    logger = setup_logging()

    try:
        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)

        # Log system information
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

        # Create results directory
        Path("results").mkdir(exist_ok=True)

        # Generate synthetic data
        logger.info("Generating synthetic data...")
        generator = SyntheticDataGenerator()
        data = generator.generate_data()

        # Train models and get predictions
        logger.info("Starting model training...")
        results = train_and_evaluate_all_models(data, logger)

        # Save results
        logger.info("Saving results...")
        with open("results/model_results.json", "w") as f:
            json.dump(results, f)

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        torch.cuda.empty_cache()
        raise


if __name__ == "__main__":
    main()