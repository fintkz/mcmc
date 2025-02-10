import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Union, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def safe_divide(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Safely divide two arrays, handling division by zero

    Args:
        a: Numerator array
        b: Denominator array
        eps: Small constant to prevent division by zero

    Returns:
        Result of a/b with zeros where b is zero
    """
    return np.divide(a, b + eps, out=np.zeros_like(a), where=b != 0)


def calculate_mape(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
) -> float:
    """Calculate Mean Absolute Percentage Error

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAPE value as percentage
    """
    # Convert to numpy if tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure arrays are flat
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    # Remove zeros from calculation
    mask = y_true != 0
    if not mask.any():
        return 0.0  # All zeros case

    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    # Calculate MAPE
    abs_percentage_error = np.abs(
        (y_true_filtered - y_pred_filtered) / y_true_filtered
    )
    mape = 100.0 * np.mean(abs_percentage_error)

    return float(mape)


def calculate_rmse(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
) -> float:
    """Calculate Root Mean Square Error

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        RMSE value
    """
    # Convert to numpy if tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure arrays are flat
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate_predictions(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
) -> Dict[str, float]:
    """Evaluate predictions using multiple metrics

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary containing metrics:
            - rmse: Root Mean Square Error
            - mape: Mean Absolute Percentage Error
    """
    return {
        "rmse": calculate_rmse(y_true, y_pred),
        "mape": calculate_mape(y_true, y_pred),
    }


def moving_average(
    data: np.ndarray, window: int, min_periods: Optional[int] = None
) -> np.ndarray:
    """Calculate moving average of a time series

    Args:
        data: Input array
        window: Window size for moving average
        min_periods: Minimum number of observations required

    Returns:
        Array of moving averages
    """
    if min_periods is None:
        min_periods = window

    cumsum = np.cumsum(np.insert(data, 0, 0))
    ma = (cumsum[window:] - cumsum[:-window]) / window

    # Handle the first window-1 elements
    result = np.empty_like(data)
    result[window - 1 :] = ma

    # Fill initial values with expanding mean
    for i in range(window - 1):
        if i + 1 >= min_periods:
            result[i] = data[: i + 1].mean()
        else:
            result[i] = np.nan

    return result


def peak_weighted_loss(y_pred, y_true, alpha=1.5, beta=0.5):
    """Custom loss function for handling demand spikes with better scaling"""
    if len(y_true.shape) == 1:
        y_true = y_true.view(-1, 1)
    if len(y_pred.shape) == 1:
        y_pred = y_pred.view(-1, 1)

    # Scale inputs to prevent exploding gradients
    y_true_scaled = (y_true - y_true.mean()) / (y_true.std() + 1e-8)
    y_pred_scaled = (y_pred - y_true.mean()) / (y_true.std() + 1e-8)

    # Base loss using scaled values
    base_loss = F.mse_loss(y_pred_scaled, y_true_scaled, reduction="none")

    # Identify peaks using scaled values
    peak_mask = (y_true_scaled > beta).float()

    # Additional weight for points where prediction is far from truth
    pred_error = torch.abs(y_pred_scaled - y_true_scaled)
    error_weight = 1.0 + torch.tanh(
        pred_error
    )  # Use tanh instead of sigmoid for better stability

    # Combine weights with smaller alpha to prevent explosion
    total_weight = 1.0 + (alpha - 1.0) * peak_mask * error_weight

    # Clip weights to prevent extreme values
    total_weight = torch.clamp(total_weight, 0.1, 5.0)

    return (base_loss * total_weight).mean()


def plot_predictions_comparison(
    y_true, predictions_dict, title="Model Comparison"
):
    """
    Plot predictions from multiple models
    predictions_dict: {"model_name": predictions}
    """
    plt.figure(figsize=(15, 6))
    x = np.arange(len(y_true))

    # Plot ground truth
    plt.plot(x, y_true, label="Actual", color="black", alpha=0.5)

    # Plot each model's predictions
    colors = ["blue", "red", "green", "purple"]
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
    plt.imshow(attention_weights, cmap="viridis", aspect="auto")
    plt.colorbar()

    if feature_names:
        plt.xticks(range(len(feature_names)), feature_names, rotation=45)
        plt.yticks(range(len(feature_names)), feature_names)

    plt.title("Attention Weights Heatmap")
    plt.tight_layout()
    return plt.gcf()


def calculate_mape_torch(
    y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Calculate Mean Absolute Percentage Error while maintaining gradients

    Args:
        y_true: True values tensor
        y_pred: Predicted values tensor
        eps: Small constant to prevent division by zero

    Returns:
        MAPE value as a tensor with gradients
    """
    # Keep tensors on their original device
    device = y_true.device

    # Mask zero values
    mask = y_true != 0

    if not mask.any():
        # Return zero tensor with grad_fn
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Calculate percentage error
    percentage_error = torch.abs(
        (y_true[mask] - y_pred[mask]) / (y_true[mask] + eps)
    )

    # Return mean error as percentage
    return 100.0 * torch.mean(percentage_error)
