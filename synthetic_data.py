import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_grocery_data(n_days=365):
    """Generate synthetic grocery sales data with known patterns."""
    
    # Create date range
    dates = [datetime.now() - timedelta(days=x) for x in range(n_days)]
    dates.reverse()
    
    # Base demand pattern
    t = np.arange(n_days)
    base_demand = 1000 + 100 * np.sin(2 * np.pi * t / 7)  # Weekly seasonality
    
    # Add various effects
    weather_effect = np.random.normal(0, 50, n_days)  # Weather variation
    special_events = np.zeros(n_days)
    special_events[np.random.choice(n_days, 20)] = 200  # Random events
    
    # Trend component
    trend = 0.5 * t
    
    # Combine components
    true_demand = base_demand + weather_effect + special_events + trend
    
    # Add noise
    observed_demand = true_demand + np.random.normal(0, 30, n_days)
    
    # Generate features
    weather_temp = 20 + 10 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 3, n_days)
    is_weekend = [d.weekday() >= 5 for d in dates]
    is_holiday = np.random.choice([0, 1], size=n_days, p=[0.95, 0.05])
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'demand': observed_demand,
        'temperature': weather_temp,
        'is_weekend': is_weekend,
        'is_holiday': is_holiday,
        'true_demand': true_demand  # For evaluation
    })
    
    return df

if __name__ == "__main__":
    df = generate_synthetic_grocery_data()
    print("Generated data shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
