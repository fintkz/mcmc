
# Retail Demand Forecasting: Model Comparison

## Motivation
Retail demand forecasting is a crucial business problem that requires balancing multiple objectives:
- Accurate predictions to optimize inventory
- Understanding uncertainty for risk management
- Interpretable results for business decisions
- Handling complex patterns including seasonality and special events

This project explores three different approaches to demand forecasting, each with its own strengths:
1. Bayesian Neural Network Ensemble - For uncertainty quantification
2. Temporal Fusion Transformer - For complex temporal patterns
3. Prophet - For interpretable seasonality components

## Problem Statement
Given historical retail demand data with various features (temperature, holidays, etc.), predict future demand while:
- Quantifying prediction uncertainty
- Capturing seasonal patterns
- Handling demand spikes
- Providing interpretable insights

## Approach
The project implements a comparative study using synthetic retail data that mimics real-world patterns:
- Base seasonal patterns (yearly, weekly)
- Special events (holidays, promotions)
- External factors (temperature, weekends)
- Random demand spikes

Each model addresses different aspects of the problem:
- Bayesian Ensemble provides uncertainty estimates
- TFT captures complex temporal dependencies
- Prophet offers interpretable decomposition of trends

## Project Structure
```
mcmc/
├── data.py           # Data generation and preprocessing
├── main.py          # Main training and evaluation script
├── models/
│   ├── bayesian_ensemble.py    # Bayesian Neural Network Ensemble
│   ├── prophet_model.py        # Facebook Prophet wrapper
│   └── temporal_fusion.py      # Temporal Fusion Transformer
├── utils.py         # Shared utilities (loss, metrics, visualization)
├── requirements.txt # Project dependencies
└── results/        # Generated visualizations and metrics
    ├── prophet_components.png
    ├── model_comparison.png
    ├── bayesian_uncertainty.png
    ├── tft_attention.png
    └── metrics.txt
```

## Setup and Usage

Run the comparison:
```bash
uv run main.py
```

The script will:
- Generate synthetic retail data
- Train all three models
- Save visualizations and metrics to 'results' directory

## Model Details

### Bayesian Neural Network Ensemble
- Ensemble of neural networks with uncertainty estimation
- Uses Monte Carlo sampling for predictions
- Provides confidence intervals for forecasts
- Custom peak-weighted loss for handling demand spikes

### Temporal Fusion Transformer (TFT)
- Attention-based architecture for time series
- Handles multiple feature types
- Captures long-range dependencies
- Provides attention visualization

### Prophet
- Decomposable time series model
- Handles multiple seasonality
- Interpretable components
- Built-in handling of holidays and events

## Results
The models are evaluated using:
- RMSE (Root Mean Square Error)
- MAPE (Mean Absolute Percentage Error)
- Visual comparison of predictions
- Uncertainty visualization (Bayesian)
- Component analysis (Prophet)
- Attention patterns (TFT)

Results are saved in the 'results' directory with:
- Comparative performance plots
- Individual model visualizations
- Numerical metrics

## Future Work
- Support for real retail data
- Additional models (LightGBM, DeepAR)
- Hyperparameter optimization
- Online learning capabilities
- Multi-store/multi-product forecasting

## GPU Support
The project is optimized for GPU usage, particularly:
- Bayesian Ensemble training
- TFT model operations
- Batch processing of large datasets

Recommended: NVIDIA GPU with CUDA support for optimal performance.