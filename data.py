import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def generate_synthetic_data(n_days=365, include_events=True):
    """
    Generate synthetic retail data with multiple patterns:
    - Base seasonal patterns (yearly, weekly)
    - Special events (holidays, promotions)
    - External factors (temperature, weekends)
    - Random demand spikes
    """
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=n_days)
    
    # Base demand patterns
    t = np.arange(n_days)
    yearly_pattern = 200 * np.sin(2 * np.pi * t / 365)
    weekly_pattern = 50 * np.sin(2 * np.pi * t / 7)
    base_demand = 1000 + yearly_pattern + weekly_pattern
    
    # Add promotion and holiday effects
    if include_events:
        # Random holidays (10 days)
        holidays = np.zeros(n_days)
        holiday_idx = np.random.choice(n_days, 10, replace=False)
        holidays[holiday_idx] = 1
        holiday_effect = holidays * np.random.uniform(200, 400, n_days)
        
        # Promotions (every 30 days on average)
        promotions = np.zeros(n_days)
        promo_idx = np.arange(0, n_days, 30) + np.random.randint(-5, 5, size=len(np.arange(0, n_days, 30)))
        promo_idx = promo_idx[(promo_idx >= 0) & (promo_idx < n_days)]
        promotions[promo_idx] = 1
        promo_effect = promotions * np.random.uniform(100, 300, n_days)
        
        base_demand += holiday_effect + promo_effect
    
    # Add noise and random spikes
    noise = np.random.normal(0, 50, n_days)
    spikes = np.random.binomial(1, 0.1, n_days) * np.random.uniform(100, 300, n_days)
    demand = base_demand + noise + spikes
    
    # Generate temperature (correlated with yearly pattern)
    temp_base = 20 + 10 * np.sin(2 * np.pi * t / 365)
    temperature = temp_base + np.random.normal(0, 2, n_days)
    
    # Calendar features
    is_weekend = [1 if d.weekday() >= 5 else 0 for d in dates]
    
    df = pd.DataFrame({
        'ds': dates,  # Prophet format
        'date': dates,  # General format
        'y': demand,  # Prophet format
        'demand': demand,  # General format
        'temperature': temperature,
        'is_weekend': is_weekend,
        'is_holiday': holidays if include_events else np.zeros(n_days),
        'is_promotion': promotions if include_events else np.zeros(n_days)
    })
    
    return df

def preprocess_data(df, include_lags=True):
    """
    Preprocess data with features suitable for different models:
    - Prophet: requires 'ds' and 'y' columns
    - TFT: can use rich feature set
    - XGBoost: can use all features
    """
    df = df.copy()
    
    if include_lags:
        # Add lag features
        df['demand_lag1'] = df['demand'].shift(1)
        df['demand_lag7'] = df['demand'].shift(7)
        
        # Rolling statistics
        for window in [7, 14, 30]:
            df[f'rolling_mean_{window}d'] = df['demand'].rolling(window).mean()
            df[f'rolling_std_{window}d'] = df['demand'].rolling(window).std()
            df[f'rolling_max_{window}d'] = df['demand'].rolling(window).max()
        
        # Trend indicators
        df['trend_7d'] = df['demand'].diff(7)
        df['trend_1d'] = df['demand'].diff(1)
    
    # Cyclic encoding of date features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    return df.dropna() if include_lags else df

def get_feature_sets():
    """Return standard feature sets for each model type"""
    prophet_features = ['ds', 'y', 'temperature', 'is_weekend', 'is_holiday', 'is_promotion']
    
    tft_features = [
        'temperature', 'is_weekend', 'is_holiday', 'is_promotion',
        'demand_lag1', 'demand_lag7', 'rolling_mean_7d', 'rolling_std_7d',
        'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos'
    ]
    
    xgb_features = [
        'temperature', 'is_weekend', 'is_holiday', 'is_promotion',
        'demand_lag1', 'demand_lag7', 'rolling_mean_7d', 'rolling_std_7d',
        'rolling_max_7d', 'trend_7d', 'trend_1d',
        'day_of_week_sin', 'day_of_week_cos'
    ]
    
    return prophet_features, tft_features, xgb_features