import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

from data import generate_synthetic_grocery_data, preprocess_data
from models.bayesian_ensemble import GPUBayesianEnsemble
from models.temporal_fusion import TFTModel

def main():
    # Generate and preprocess data
    print("Generating synthetic data...")
    df = generate_synthetic_grocery_data()
    X, y, scaler_y, feature_names = preprocess_data(df)

    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Train models
    print("\nTraining Bayesian Ensemble...")
    bayes_model = GPUBayesianEnsemble(input_dim=X.shape[1])
    bayes_model.train(X_train, y_train, epochs=400)

    print("\nTraining Temporal Fusion Transformer...")
    tft_model = TFTModel(num_features=X.shape[1])
    tft_model.train(X_train, y_train)

    print("\nTraining XGBoost...")
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

    params = {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'device': 'cuda',
        'max_depth': 6,
        'learning_rate': 0.1,
        'eval_metric': 'rmse'
    }

    xgb_model = xgb.train(params, dtrain, num_boost_round=100)

    # Get predictions
    bayes_mean, bayes_std = bayes_model.predict(X_test)
    tft_pred = tft_model.predict(X_test)
    xgb_pred = xgb_model.predict(dtest)

    # Inverse transform predictions
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    xgb_pred_orig = scaler_y.inverse_transform(xgb_pred.reshape(-1, 1)).flatten()
    bayes_mean_orig = scaler_y.inverse_transform(bayes_mean.reshape(-1, 1)).flatten()
    tft_pred_orig = scaler_y.inverse_transform(tft_pred.reshape(-1, 1)).flatten()
    bayes_std_orig = bayes_std * scaler_y.scale_

    # Calculate MSE
    bayes_mse = np.mean((bayes_mean_orig - y_test_orig) ** 2)
    xgb_mse = np.mean((xgb_pred_orig - y_test_orig) ** 2)
    tft_mse = np.mean((tft_pred_orig - y_test_orig) ** 2)

    print(f"\nBayesian Ensemble MSE: {bayes_mse:.2f}")
    print(f"XGBoost MSE: {xgb_mse:.2f}")
    print(f"TFT MSE: {tft_mse:.2f}")

    # Plot results
    plot_results(y_test_orig, xgb_pred_orig, bayes_mean_orig, 
                tft_pred_orig, bayes_std_orig)

def plot_results(y_test, xgb_pred, bayes_mean, tft_pred, bayes_std):
    plt.figure(figsize=(15, 7))
    x_axis = np.arange(len(y_test))

    plt.plot(x_axis, y_test, label='Ground Truth', color='black', alpha=0.5)
    plt.plot(x_axis, xgb_pred, label='XGBoost', color='blue', alpha=0.7)
    plt.plot(x_axis, bayes_mean, label='Bayesian Ensemble', color='red', alpha=0.7)
    plt.plot(x_axis, tft_pred, label='TFT', color='green', alpha=0.7)

    plt.fill_between(x_axis, 
                     bayes_mean - 2*bayes_std, 
                     bayes_mean + 2*bayes_std,
                     color='red', alpha=0.2, 
                     label='Bayesian 95% CI')

    plt.xlabel('Time Steps')
    plt.ylabel('Demand')
    plt.title('Model Predictions vs Ground Truth')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig('predictions_comparison.png')
    plt.close()

if __name__ == "__main__":
    main()