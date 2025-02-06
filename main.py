import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from bayesian_ensemble import GPUBayesianEnsemble
from temporal_fusion import TFTModel
from enhanced_preprocessing import enhanced_preprocess_data
import xgboost as xgb

# Generate data
print("Generating synthetic data...")
from synthetic_data import generate_synthetic_grocery_data
df = generate_synthetic_grocery_data()

# Enhanced preprocessing
print("\nPreprocessing data with enhanced features...")
X, y, scaler_y, feature_names = enhanced_preprocess_data(df)

# Split data
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train Bayesian Ensemble
print("\nTraining Bayesian Ensemble...")
bayes_model = GPUBayesianEnsemble(input_dim=X.shape[1])
bayes_model.train(X_train, y_train, epochs=400)  # Doubled epochs to 400

# Train TFT
print("\nTraining Temporal Fusion Transformer...")
tft_model = TFTModel(num_features=X.shape[1])
tft_model.train(X_train, y_train)

# Train XGBoost
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

# Get and print feature importance for XGBoost
importance_scores = xgb_model.get_score(importance_type='gain')
importance_sorted = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
print("\nTop 10 Most Important Features:")
for feat, imp in importance_sorted[:10]:
    print(f"{feat}: {imp:.4f}")

# Plotting
plt.figure(figsize=(15, 7))
x_axis = np.arange(len(y_test_orig))

plt.plot(x_axis, y_test_orig, label='Ground Truth', color='black', alpha=0.5)
plt.plot(x_axis, xgb_pred_orig, label='XGBoost', color='blue', alpha=0.7)
plt.plot(x_axis, bayes_mean_orig, label='Bayesian Ensemble', color='red', alpha=0.7)
plt.plot(x_axis, tft_pred_orig, label='TFT', color='green', alpha=0.7)

# Add uncertainty bands for Bayesian predictions
plt.fill_between(x_axis, 
                 bayes_mean_orig - 2*bayes_std_orig, 
                 bayes_mean_orig + 2*bayes_std_orig,
                 color='red', alpha=0.2, 
                 label='Bayesian 95% CI')

plt.xlabel('Time Steps')
plt.ylabel('Demand')
plt.title('Model Predictions vs Ground Truth')
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig('predictions_comparison_v2.png')
plt.close()

print("\nVisualization saved as 'predictions_comparison_v2.png'")
