import numpy as np
import pandas as pd
import json
from itertools import combinations
from pathlib import Path
import torch.multiprocessing as mp
from functools import partial
import torch
from data import SyntheticDataGenerator
from models.prophet_model import ProphetModel
from models.temporal_fusion import TFTModel
from models.bayesian_ensemble import GPUBayesianEnsemble
from utils import evaluate_predictions
import os
import logging
from datetime import datetime

def setup_logging():
    """Setup logging configuration"""
    log_dir = Path('results')
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'work_log_{timestamp}.txt'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    return logging.getLogger(__name__)

def create_temporal_features(df):
    """Create temporal features for the dataset"""
    # Basic time features
    df['day_of_year'] = df['ds'].dt.dayofyear
    df['day_of_week'] = df['ds'].dt.dayofweek
    df['month'] = df['ds'].dt.month
    df['week_of_year'] = df['ds'].dt.isocalendar().week
    
    # Cyclical encoding of temporal features
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year']/365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year']/365)
    df['week_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['week_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    # Create lag features
    for lag in [1, 7, 14, 30]:  # Common time series lags
        df[f'lag_{lag}'] = df['y'].shift(lag)
    
    # Fill NaN values in lag features with 0
    df = df.fillna(0)
    
    return df

def prepare_feature_combinations(df, features):
    """Generate all possible feature combinations and corresponding datasets"""
    all_combinations = []
    for r in range(len(features) + 1):
        all_combinations.extend(combinations(features, r))
    return list(all_combinations)

def train_combination(combo, gpu_id, features, temporal_features, df, logger):
    """Train models for a specific feature combination"""
    try:
        # Set GPU device
        torch.cuda.set_device(gpu_id)
        
        combo_name = '_'.join(combo) if combo else 'baseline'
        logger.info(f"\nTraining models for combination: {combo_name} on GPU {gpu_id}")
        
        df_subset = df.copy()
        # For missing features, set their values to 0
        missing_features = set(features) - set(combo)
        for feature in missing_features:
            df_subset[feature] = 0
        
        # Prepare feature matrix with temporal features
        X = np.hstack([
            df_subset[list(features)].values,
            df_subset[temporal_features].values
        ])
        y = df_subset['y'].values
        
        # Prophet
        logger.info(f"Training Prophet for {combo_name}")
        prophet_model = ProphetModel()
        prophet_df = pd.DataFrame({
            'ds': df_subset['ds'],
            'y': df_subset['y']
        })
        for feature in combo:
            prophet_df[feature] = df_subset[feature]
        prophet_model.add_external_features(list(combo))
        prophet_model.fit(prophet_df)
        prophet_forecast = prophet_model.predict(prophet_df)
        
        # TFT
        logger.info(f"Training TFT for {combo_name}")
        tft_model = TFTModel(
            num_features=X.shape[1],
            seq_length=30,
            batch_size=256,
            num_workers=4,
            device=f'cuda:{gpu_id}'
        )
        tft_model.train(X, y)
        tft_preds = tft_model.predict(X)
        
        # Bayesian
        logger.info(f"Training Bayesian for {combo_name}")
        bayesian_model = GPUBayesianEnsemble(
            input_dim=X.shape[1],
            device=f'cuda:{gpu_id}',
            batch_size=256,
            num_workers=4
        )
        bayesian_model.train(X, y)
        bayesian_mean, bayesian_std = bayesian_model.predict(X)
        
        # Store results
        result = {
            'features': list(combo),
            'prophet': {
                'yhat': prophet_forecast['yhat'].tolist(),
                'metrics': evaluate_predictions(y, prophet_forecast['yhat'])
            },
            'tft': {
                'yhat': tft_preds.tolist(),
                'metrics': evaluate_predictions(y, tft_preds)
            },
            'bayesian': {
                'yhat': bayesian_mean.tolist(),
                'uncertainty': bayesian_std.tolist(),
                'metrics': evaluate_predictions(y, bayesian_mean)
            }
        }
        
        # Log metrics
        logger.info(f"Results for {combo_name} on GPU {gpu_id}:")
        logger.info(f"Prophet MAPE: {result['prophet']['metrics']['mape']:.2f}%")
        logger.info(f"TFT MAPE: {result['tft']['metrics']['mape']:.2f}%")
        logger.info(f"Bayesian MAPE: {result['bayesian']['metrics']['mape']:.2f}%")
        
        return combo_name, result
        
    except Exception as e:
        logger.error(f"Error processing combination {combo_name} on GPU {gpu_id}: {str(e)}")
        raise

def train_and_evaluate_all_models(df, feature_dates, logger):
    """Train all models with different feature combinations using multiple GPUs"""
    features = ['promotions_active', 'weather_event', 'sports_event', 
                'school_term', 'holiday']
    
    # Add temporal features
    logger.info("Creating temporal features...")
    df = create_temporal_features(df)
    temporal_features = [
        'day_sin', 'day_cos', 'week_sin', 'week_cos',
        'month_sin', 'month_cos', 'lag_1', 'lag_7',
        'lag_14', 'lag_30'
    ]
    
    # Get feature combinations
    combinations = prepare_feature_combinations(df, features)
    
    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    logger.info(f"Training using {num_gpus} GPUs")
    
    # Initialize multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # Create partial function with fixed arguments
    train_func = partial(
        train_combination,
        features=features,
        temporal_features=temporal_features,
        df=df,
        logger=logger
    )
    
    # Distribute combinations across GPUs
    results = {'feature_dates': feature_dates, 'actual': df['y'].tolist(), 'predictions': {}}
    
    # Create process pool
    with mp.Pool(num_gpus) as pool:
        # Create arguments list with only combo and gpu_id
        args = [(combo, i % num_gpus) for i, combo in enumerate(combinations)]
        
        # Map combinations to processes using only combo and gpu_id
        combo_results = pool.starmap(train_func, args)
        
        # Collect results
        for combo_name, result in combo_results:
            results['predictions'][combo_name] = result
    
    return results

def main():
    # Setup logging
    logger = setup_logging()
    
    try:
        # Enable anomaly detection for better error messages
        torch.autograd.set_detect_anomaly(True)
        
        # Log system information
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Create results directory
        Path('results').mkdir(exist_ok=True)
        
        # Generate synthetic data
        logger.info("Generating synthetic data...")
        generator = SyntheticDataGenerator()
        df, feature_dates = generator.generate_data()
        
        # Train models and get predictions
        logger.info("Starting model training...")
        results = train_and_evaluate_all_models(df, feature_dates, logger)
        
        # Save results
        logger.info("Saving results...")
        with open('results/model_results.json', 'w') as f:
            json.dump(results, f)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise
    
if __name__ == "__main__":
    main()