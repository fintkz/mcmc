import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import shutil
from pathlib import Path

from data import generate_synthetic_data, preprocess_data, get_feature_sets
from models.bayesian_ensemble import GPUBayesianEnsemble
from models.temporal_fusion import TFTModel
from models.prophet_model import ProphetModel
import utils

def setup_results_dir(dir_name='results'):
    """Setup results directory, removing if it exists and recreating it"""
    results_dir = Path(dir_name)
    
    # Remove directory if it exists
    if results_dir.exists():
        shutil.rmtree(results_dir)
    
    # Create fresh directory
    results_dir.mkdir(exist_ok=True)
    return results_dir

def main():
    # Setup results directory
    results_dir = setup_results_dir()
    
    # Generate data with rich patterns
    print("Generating synthetic data...")
    df = generate_synthetic_data(n_days=365)
    df_processed = preprocess_data(df)
    
    # Get feature sets for each model
    prophet_features, tft_features, xgb_features = get_feature_sets()
    
    # Train/test split
    train_size = int(0.8 * len(df_processed))
    df_train = df_processed.iloc[:train_size]
    df_test = df_processed.iloc[train_size:]
    
    # 1. Prophet - Good for interpretable seasonality
    print("\nTraining Prophet...")
    prophet = ProphetModel()
    prophet.add_external_features(['temperature', 'is_weekend', 'is_holiday', 'is_promotion'])
    prophet.fit(df_train[prophet_features])
    prophet_forecast = prophet.predict(df_test[['ds'] + [f for f in prophet_features if f not in ['ds', 'y']]])
    
    # Plot Prophet components
    prophet.plot_components(prophet_forecast)
    plt.savefig(results_dir / 'prophet_components.png')
    plt.close()
    
    # 2. TFT - Good for complex patterns
    print("\nTraining TFT...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train = scaler_X.fit_transform(df_train[tft_features])
    y_train = scaler_y.fit_transform(df_train[['demand']]).flatten()
    X_test = scaler_X.transform(df_test[tft_features])
    y_test = scaler_y.transform(df_test[['demand']]).flatten()
    
    tft_model = TFTModel(num_features=len(tft_features), device='cuda')
    tft_model.train(X_train, y_train)
    tft_pred = tft_model.predict(X_test)
    tft_pred = scaler_y.inverse_transform(tft_pred.reshape(-1, 1)).flatten()
    
    # 3. Bayesian Ensemble - Good for uncertainty
    print("\nTraining Bayesian Ensemble...")
    bayes_model = GPUBayesianEnsemble(input_dim=len(xgb_features), n_models=15, device='cuda')
    
    X_train_bayes = df_train[xgb_features].values
    y_train_bayes = df_train['demand'].values
    X_test_bayes = df_test[xgb_features].values
    y_test_bayes = df_test['demand'].values
    
    bayes_model.train(
        X_train_bayes, 
        y_train_bayes,
        epochs=1200,
        batch_size=128,
        num_samples=15
    )
    bayes_mean, bayes_std = bayes_model.predict(X_test_bayes)
    
    # Evaluate all models
    actual = df_test['demand'].values
    predictions = {
        'Prophet': prophet_forecast['yhat'].values,
        'TFT': tft_pred,
        'Bayesian': bayes_mean
    }
    
    # Save metrics to file
    with open(results_dir / 'metrics.txt', 'w') as f:
        for name, preds in predictions.items():
            metrics = utils.evaluate_predictions(actual, preds, name)
            f.write(f"\n{name} Performance:\n")
            f.write(f"RMSE: {metrics['rmse']:.2f}\n")
            f.write(f"MAPE: {metrics['mape']:.2f}%\n")
    
    # Plot comparison
    fig = utils.plot_predictions_comparison(
        actual, 
        predictions,
        title="Model Comparison on Test Set"
    )
    fig.savefig(results_dir / 'model_comparison.png')
    plt.close()
    
    # Plot Bayesian uncertainty
    plt.figure(figsize=(15, 6))
    plt.plot(actual, label='Actual', color='black', alpha=0.5)
    plt.plot(bayes_mean, label='Bayesian Prediction', color='red', alpha=0.7)
    plt.fill_between(
        range(len(bayes_mean)),
        bayes_mean - 2*bayes_std,
        bayes_mean + 2*bayes_std,
        color='red',
        alpha=0.2,
        label='Bayesian 95% CI'
    )
    plt.title("Bayesian Ensemble Predictions with Uncertainty")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(results_dir / 'bayesian_uncertainty.png')
    plt.close()
    
    # Plot TFT attention (if available)
    if hasattr(tft_model.model, 'attention_weights'):
        attention_fig = utils.plot_attention_heatmap(
            tft_model.model.attention_weights[0],  # first head
            feature_names=tft_features
        )
        attention_fig.savefig(results_dir / 'tft_attention.png')
        plt.close()

    print("\nResults saved in 'results' directory:")
    for file in sorted(os.listdir(results_dir)):
        print(f"- {file}")

if __name__ == "__main__":
    main()