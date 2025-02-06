from bayesian_ensemble import GPUBayesianEnsemble, preprocess_data_gpu
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from synthetic_data import generate_synthetic_grocery_data
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Generate data
print("Generating synthetic data...")
df = generate_synthetic_grocery_data()
X, y, scaler_y = preprocess_data_gpu(df)

# Split data
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train Bayesian Ensemble
print("\nTraining Bayesian Ensemble...")
bayes_model = GPUBayesianEnsemble(input_dim=X.shape[1])
bayes_model.train(X_train, y_train, epochs=100)

# Train XGBoost
print("\nTraining XGBoost...")
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'reg:squarederror',
    'tree_method': 'hist',  # Updated from gpu_hist
    'device': 'cuda',      # Updated GPU specification
    'max_depth': 6,
    'learning_rate': 0.1,
    'eval_metric': 'rmse'
}

xgb_model = xgb.train(params, dtrain, num_boost_round=100)
xgb_pred = xgb_model.predict(dtest)

# Get Bayesian predictions
bayes_mean, bayes_std = bayes_model.predict(X_test)

# Inverse transform predictions
y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
xgb_pred_orig = scaler_y.inverse_transform(xgb_pred.reshape(-1, 1)).flatten()
bayes_mean_orig = scaler_y.inverse_transform(bayes_mean.reshape(-1, 1)).flatten()
bayes_std_orig = bayes_std * scaler_y.scale_

# Calculate MSE on original scale
bayes_mse = mean_squared_error(y_test_orig, bayes_mean_orig)
xgb_mse = mean_squared_error(y_test_orig, xgb_pred_orig)

print(f"\nBayesian Ensemble MSE: {bayes_mse:.2f}")
print(f"XGBoost MSE: {xgb_mse:.2f}")

# Plotting
plt.figure(figsize=(15, 7))
x_axis = np.arange(len(y_test_orig))

plt.plot(x_axis, y_test_orig, label='Ground Truth', color='black', alpha=0.5)
plt.plot(x_axis, xgb_pred_orig, label='XGBoost', color='blue', alpha=0.7)
plt.plot(x_axis, bayes_mean_orig, label='Bayesian Ensemble', color='red', alpha=0.7)

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

plt.savefig('predictions_comparison.png')
plt.close()

print("\nVisualization saved as 'predictions_comparison.png'")
