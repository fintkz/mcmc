import torch
import torch.nn.functional as F

def peak_weighted_loss(y_pred, y_true, alpha=2.0, beta=0.5):
    """
    Custom loss function that puts more emphasis on peaks and hard cases
    
    Args:
        y_pred: Predicted values
        y_true: True values
        alpha: Weight multiplier for peaks (default: 2.0)
        beta: Threshold for peaks in std units (default: 0.5)
    """
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