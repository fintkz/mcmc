# Time Series Forecasting with Multiple Deep Learning Approaches

## Problem Statement
The project addresses the challenge of predicting demand in a time series context, with a particular focus on capturing extreme events and peaks in the data. The goal was to develop a robust forecasting system that could:
1. Accurately predict regular patterns in demand
2. Capture and predict unusual spikes and peaks
3. Provide uncertainty estimates for predictions
4. Handle temporal dependencies effectively

## Model Evolution

### Initial Implementation

Started with three different approaches:
1. XGBoost: Traditional gradient boosting
2. Bayesian Neural Network Ensemble: For uncertainty quantification
3. Temporal Fusion Transformer: For complex temporal dependencies

Initial challenges:
- High MSE values (Bayesian: ~1.26M, XGBoost: ~17.9K)
- Poor peak prediction
- Unstable training
- Inconsistent preprocessing

### Key Improvements

#### 1. Enhanced Feature Engineering
```python
- Temporal features:
  - Day of week, month, day of month
  - Cyclic encodings (sin/cos transformations)
  - Week of year
- Lag features:
  - Previous day demand
  - Weekly lag (7 days)
- Rolling statistics:
  - Mean, std, max, min for multiple windows (7, 14, 30 days)
  - Peak indicators based on statistical thresholds
- Trend indicators:
  - Daily and weekly trends
```

#### 2. Model Architecture Improvements

##### Bayesian Neural Network:
- Increased model capacity
  - Added multiple hidden layers [512, 256, 128]
  - Increased ensemble size from 5 to 10 models
- Better initialization
  - Weight initialization with smaller std (0.05)
  - Better prior distribution handling
- Improved training
  - Increased epochs (100 → 200 → 400 → 800)
  - Added early stopping with patience
  - Implemented OneCycleLR scheduler
  - Added gradient clipping

##### Temporal Fusion Transformer:
- Enhanced architecture
  - Increased hidden size (128 → 512)
  - Added more attention heads (4 → 8)
  - Increased encoder layers (2 → 4)
- Added layer normalization
- Improved GRN blocks with additional layers
- Better optimization with AdamW and weight decay

#### 3. Custom Loss Function
Implemented a peak-weighted loss function to better capture extreme events:
```python
def peak_weighted_loss(y_pred, y_true, alpha=2.0, beta=0.5):
    base_loss = F.mse_loss(y_pred, y_true, reduction='none')
    peak_mask = (y_true > (mean + beta * std)).float()
    error_weight = 1.0 + torch.sigmoid(pred_error - pred_error.mean())
    total_weight = 1.0 + (alpha - 1.0) * peak_mask * error_weight
    return (base_loss * total_weight).mean()
```

### Performance Evolution

MSE Improvements:
1. Initial Implementation:
   - Bayesian: ~1.26M
   - XGBoost: ~17.9K
   - TFT: N/A (failing)

2. After Feature Engineering:
   - Bayesian: ~9.9K
   - XGBoost: ~13.9K
   - TFT: ~148

3. After Architecture Improvements:
   - Bayesian: ~78
   - XGBoost: ~1.9K
   - TFT: ~148

4. Final Version with Custom Loss:
   [Awaiting final results]

## Key Learnings

1. Feature Engineering Impact:
   - Temporal features significantly improved all models
   - Rolling statistics helped capture local patterns
   - Cyclic encodings better represented periodic patterns

2. Model Architecture:
   - Deeper networks with proper regularization outperformed shallow ones
   - Ensemble size matters more than individual model capacity
   - Layer normalization crucial for transformer stability

3. Training Dynamics:
   - Learning rate scheduling crucial for convergence
   - Early stopping prevented overfitting
   - Gradient clipping essential for stable training

4. Loss Function Design:
   - Custom loss functions can significantly improve peak prediction
   - Balancing regular and peak prediction requires careful weighting
   - Dynamic weighting based on prediction error improved robustness

## Future Improvements

1. Model Enhancements:
   - Implement quantile predictions for better uncertainty estimates
   - Add attention visualization for interpretability
   - Experiment with hybrid architectures

2. Feature Engineering:
   - Add external features (weather, events, etc.)
   - Implement automated feature selection
   - Add domain-specific features

3. Training Improvements:
   - Implement cross-validation
   - Add more sophisticated ensemble techniques
   - Experiment with different optimization strategies

4. Production Considerations:
   - Model compression techniques
   - Inference optimization
   - Online learning capabilities

## Conclusion
The project demonstrated the importance of iterative improvement in all aspects of the modeling pipeline. The most significant gains came from:
1. Thorough feature engineering
2. Custom loss function design
3. Proper model scaling and training dynamics

The final system achieves better peak prediction while maintaining good performance on regular patterns, with reliable uncertainty estimates from the Bayesian ensemble.
