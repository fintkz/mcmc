import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def generate_synthetic_grocery_data(n_days=365):
    """Generate synthetic grocery store demand data"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=n_days)
    
    # Base demand with weekly and yearly seasonality
    t = np.arange(n_days)
    base_demand = 1000 + \
                  200 * np.sin(2 * np.pi * t / 365) + \
                  50 * np.sin(2 * np.pi * t / 7)  # Weekly seasonality
    
    # Add random noise and spikes
    noise = np.random.normal(0, 50, n_days)
    spikes = np.random.binomial(1, 0.1, n_days) * np.random.uniform(100, 300, n_days)
    demand = base_demand + noise + spikes
    
    # Generate temperature data (correlated with demand)
    temp_base = 20 + 10 * np.sin(2 * np.pi * t / 365)  # Yearly temperature pattern
    temperature = temp_base + np.random.normal(0, 2, n_days)
    
    # Create weekend and holiday flags
    is_weekend = [1 if d.weekday() >= 5 else 0 for d in dates]
    
    # Generate some random holidays
    holidays = np.zeros(n_days)
    n_holidays = 10
    holiday_idx = np.random.choice(n_days, n_holidays, replace=False)
    holidays[holiday_idx] = 1
    
    df = pd.DataFrame({
        'date': dates,
        'demand': demand,
        'temperature': temperature,
        'is_weekend': is_weekend,
        'is_holiday': holidays
    })
    
    return df

def preprocess_data(df):
    """
    Preprocess data with temporal and statistical features
    """
    df = df.copy()
    
    # Basic temporal features
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    
    # Lag features
    df['demand_lag1'] = df['demand'].shift(1)
    df['demand_lag7'] = df['demand'].shift(7)
    
    # Rolling statistics
    for window in [7, 14]:
        df[f'rolling_mean_{window}d'] = df['demand'].rolling(window).mean()
        df[f'rolling_max_{window}d'] = df['demand'].rolling(window).max()
        df[f'rolling_min_{window}d'] = df['demand'].rolling(window).min()
    
    # Peak indicators and trends
    df['peak_indicator'] = (
        df['demand'] > (df['demand'].rolling(7).mean() + df['demand'].rolling(7).std())
    ).astype(int)
    df['trend_7d'] = df['demand'].diff(7)
    df['trend_1d'] = df['demand'].diff(1)
    
    # Cyclic encoding
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    
    df = df.dropna()
    
    feature_columns = [
        'temperature', 'is_weekend', 'is_holiday',
        'demand_lag1', 'demand_lag7',
        'rolling_mean_7d', 'rolling_max_7d', 'rolling_min_7d',
        'rolling_max_14d', 'peak_indicator', 'trend_7d', 'trend_1d',
        'day_of_week_sin', 'day_of_week_cos'
    ]
    
    X = df[feature_columns].values
    y = df['demand'].values
    
    # Scale features and target
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    return X_scaled, y_scaled, scaler_y, feature_columns