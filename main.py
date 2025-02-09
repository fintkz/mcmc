import numpy as np
import pandas as pd
import json
from itertools import combinations
from pathlib import Path
from data import SyntheticDataGenerator
from models.prophet_model import ProphetModel
from models.temporal_fusion import TFTModel
from models.bayesian_ensemble import GPUBayesianEnsemble
from utils import evaluate_predictions

def prepare_feature_combinations(df, features):
    """Generate all possible feature combinations and corresponding datasets"""
    all_combinations = []
    for r in range(len(features) + 1):
        all_combinations.extend(combinations(features, r))
    
    datasets = {}
    for combo in all_combinations:
        df_subset = df.copy()
        
        # For missing features, set their values to 0
        missing_features = set(features) - set(combo)
        for feature in missing_features:
            df_subset[feature] = 0
            
        datasets[combo] = df_subset
        
    return datasets

def train_and_evaluate_all_models(df, feature_dates):
    """Train all models with different feature combinations"""
    features = ['promotions_active', 'weather_event', 'sports_event', 
                'school_term', 'holiday']
    
    # Prepare datasets for all feature combinations
    datasets = prepare_feature_combinations(df, features)
    
    results = {
        'feature_dates': feature_dates,
        'actual': df['y'].tolist(),
        'predictions': {}
    }
    
    # Train models on each feature combination
    for combo, df_subset in datasets.items():
        combo_name = '_'.join(combo) if combo else 'baseline'
        print(f"\nTraining models for combination: {combo_name}")
        
        # Prepare data
        X = df_subset[features].values
        y = df_subset['y'].values
        
        # Prophet
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
        tft_model = TFTModel(num_features=len(features))
        tft_model.train(X, y)
        tft_preds = tft_model.predict(X)
        
        # Bayesian
        bayesian_model = GPUBayesianEnsemble(input_dim=len(features))
        bayesian_model.train(X, y)
        bayesian_mean, bayesian_std = bayesian_model.predict(X)
        
        # Store results
        results['predictions'][combo_name] = {
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
        
    return results

def main():
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    # Generate synthetic data
    print("Generating synthetic data...")
    generator = SyntheticDataGenerator()
    df, feature_dates = generator.generate_data()
    
    # Train models and get predictions
    print("Training models...")
    results = train_and_evaluate_all_models(df, feature_dates)
    
    # Save results
    print("Saving results...")
    with open('results/model_results.json', 'w') as f:
        json.dump(results, f)
        
    print("Done! Results saved in results/model_results.json")

if __name__ == "__main__":
    main()