import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

def peak_weighted_loss(y_pred, y_true, alpha=2.0, beta=0.5):
    """Custom loss function for handling demand spikes"""
    if len(y_true.shape) == 1:
        y_true = y_true.view(-1, 1)
    if len(y_pred.shape) == 1:
        y_pred = y_pred.view(-1, 1)
        
    base_loss = F.mse_loss(y_pred, y_true, reduction='none')
    
    # Identify peaks and difficult points
    mean = y_true.mean()
    std = y_true.std()
    peak_mask = (y_true > (mean + beta * std)).float()
    
    # Additional weight for points where prediction is far from truth
    pred_error = torch.abs(y_pred - y_true)
    error_weight = 1.0 + torch.sigmoid(pred_error - pred_error.mean())
    
    # Combine weights
    total_weight = 1.0 + (alpha - 1.0) * peak_mask * error_weight
    
    return (base_loss * total_weight).mean()

def evaluate_predictions(y_true, y_pred, model_name=""):
    """Calculate multiple error metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    print(f"\n{model_name} Performance:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    return {"rmse": rmse, "mape": mape}

def plot_predictions_comparison(y_true, predictions_dict, title="Model Comparison"):
    """
    Plot predictions from multiple models
    predictions_dict: {"model_name": predictions}
    """
    plt.figure(figsize=(15, 6))
    x = np.arange(len(y_true))
    
    # Plot ground truth
    plt.plot(x, y_true, label='Actual', color='black', alpha=0.5)
    
    # Plot each model's predictions
    colors = ['blue', 'red', 'green', 'purple']
    for (name, preds), color in zip(predictions_dict.items(), colors):
        plt.plot(x, preds, label=name, color=color, alpha=0.7)
    
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def plot_attention_heatmap(attention_weights, feature_names=None):
    """Plot attention weights heatmap"""
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights, cmap='viridis', aspect='auto')
    plt.colorbar()
    
    if feature_names:
        plt.xticks(range(len(feature_names)), feature_names, rotation=45)
        plt.yticks(range(len(feature_names)), feature_names)
    
    plt.title("Attention Weights Heatmap")
    plt.tight_layout()
    return plt.gcf()