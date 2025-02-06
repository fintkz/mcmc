import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def enhanced_preprocess_data(df):
    """
    Enhanced preprocessing with temporal and statistical features
    """
    # Create copy to avoid modifying original
    df = df.copy()
    
    # Basic temporal features
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day_of_month'] = df['date'].dt.day
    df['week_of_year'] = df['date'].dt.isocalendar().week
    
    # Lag features
    df['demand_lag1'] = df['demand'].shift(1)
    df['demand_lag7'] = df['demand'].shift(7)  # Weekly lag
    
    # Rolling statistics
    for window in [7, 14, 30]:
        df[f'rolling_mean_{window}d'] = df['demand'].rolling(window).mean()
        df[f'rolling_std_{window}d'] = df['demand'].rolling(window).std()
        df[f'rolling_max_{window}d'] = df['demand'].rolling(window).max()
        df[f'rolling_min_{window}d'] = df['demand'].rolling(window).min()
    
    # Peak indicators
    df['peak_indicator'] = (
        df['demand'] > (df['demand'].rolling(7).mean() + df['demand'].rolling(7).std())
    ).astype(int)
    
    # Trend indicators
    df['trend_7d'] = df['demand'].diff(7)
    df['trend_1d'] = df['demand'].diff(1)
    
    # Cyclic encoding of temporal features
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    
    # Drop rows with NaN from rolling calculations
    df = df.dropna()
    
    # Select features for modeling
    feature_columns = [
        'temperature', 'is_weekend', 'is_holiday',
        'demand_lag1', 'demand_lag7',
        'rolling_mean_7d', 'rolling_std_7d', 'rolling_max_7d', 'rolling_min_7d',
        'rolling_mean_14d', 'rolling_max_14d',
        'peak_indicator', 'trend_7d', 'trend_1d',
        'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos'
    ]
    
    X = df[feature_columns].values
    y = df['demand'].values
    
    # Scale features and target
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    return X_scaled, y_scaled, scaler_y, feature_columns

def get_feature_importance(model, feature_names):
    """
    Get feature importance from XGBoost model
    """
    importance_dict = {
        name: score for name, score in zip(feature_names, model.feature_importances_)
    }
    importance_sorted = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    return importance_sorted
