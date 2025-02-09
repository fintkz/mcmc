import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple, NamedTuple
from dataclasses import dataclass


@dataclass
class DatasetFeatures:
    """Container for dataset features with named tensors"""
    features: torch.Tensor  # [time, features]
    temporal: torch.Tensor  # [time, temporal_features]
    target: torch.Tensor    # [time]
    dates: pd.Series
    feature_dates: Dict


class SyntheticDataGenerator:
    def __init__(self, start_date: str = "2023-01-01", periods: int = 365):
        """Initialize data generator with parameters"""
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.periods = periods
        self.feature_names = [
            'promotions_active',
            'weather_event',
            'sports_event',
            'school_term',
            'holiday'
        ]
        self.temporal_feature_names = [
            'day_sin', 'day_cos',
            'week_sin', 'week_cos',
            'month_sin', 'month_cos'
        ]

    def generate_base_demand(self) -> torch.Tensor:
        """Generate base demand with trend and seasonality"""
        time = torch.arange(self.periods, dtype=torch.float32)
        time = time.refine_names('time')

        # Trend
        trend = 1000 + time.rename(None) * 0.5

        # Yearly seasonality
        yearly = 200 * torch.sin(2 * np.pi * time.rename(None) / 365)

        # Weekly seasonality
        weekly = 50 * torch.sin(2 * np.pi * time.rename(None) / 7)

        base_demand = (trend + yearly + weekly).refine_names('time')
        return base_demand

    def add_promotions(self, base_demand: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add promotion effects with named tensors"""
        # Generate random promotion dates (about 2 per month)
        promo_dates = torch.sort(torch.randperm(self.periods)[:24])[0]
        
        # Effect lasts for 3-5 days with peak on second day
        promo_effect = torch.zeros(self.periods).refine_names('time')
        promo_indicator = torch.zeros(self.periods).refine_names('time')
        
        for date in promo_dates:
            duration = torch.randint(3, 6, (1,)).item()
            effect_pattern = torch.tensor([0.5, 1.0, 0.7, 0.3, 0.1][:duration])
            end_idx = min(date.item() + duration, self.periods)
            promo_effect.rename(None)[date:end_idx] += effect_pattern[:end_idx - date]
            promo_indicator.rename(None)[date] = 1

        return promo_effect * 300, promo_indicator

    def add_weather_effects(self, base_demand: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add weather effects with named tensors"""
        weather_dates = torch.sort(torch.randperm(self.periods)[:30])[0]
        
        weather_effect = torch.zeros(self.periods).refine_names('time')
        weather_indicator = torch.zeros(self.periods).refine_names('time')
        
        for date in weather_dates:
            duration = torch.randint(1, 4, (1,)).item()
            effect = torch.randn(1).item() * 0.2  # Random effect between -0.2 and 0.2
            end_idx = min(date.item() + duration, self.periods)
            weather_effect.rename(None)[date:end_idx] = effect
            weather_indicator.rename(None)[date] = 1

        return base_demand * weather_effect, weather_indicator

    def create_temporal_features(self, dates: pd.Series) -> torch.Tensor:
        """Create temporal features with named dimensions"""
        time = torch.arange(self.periods, dtype=torch.float32)
        time = time.refine_names('time')
        
        # Create cyclical features
        day_of_year = torch.tensor(dates.dt.dayofyear.values, dtype=torch.float32)
        week_of_year = torch.tensor(dates.dt.isocalendar().week.values, dtype=torch.float32)
        month = torch.tensor(dates.dt.month.values, dtype=torch.float32)
        
        # Stack all temporal features
        temporal_features = torch.stack([
            torch.sin(2 * np.pi * day_of_year / 365),
            torch.cos(2 * np.pi * day_of_year / 365),
            torch.sin(2 * np.pi * week_of_year / 52),
            torch.cos(2 * np.pi * week_of_year / 52),
            torch.sin(2 * np.pi * month / 12),
            torch.cos(2 * np.pi * month / 12)
        ], dim='temporal_features')
        
        return temporal_features.align_to('time', 'temporal_features')

    def generate_data(self) -> DatasetFeatures:
        """Generate complete synthetic dataset with named tensors"""
        if self.periods < 365:
            raise ValueError("Periods should be at least 365 for meaningful seasonal patterns")

        # Generate base demand
        base_demand = self.generate_base_demand()
        
        # Validate base demand
        if torch.any(torch.isnan(base_demand)) or torch.any(torch.isinf(base_demand)):
            raise ValueError("Invalid values in base demand generation")

        try:
            # Generate all effects and their indicators
            promo_effect, promo_indicator = self.add_promotions(base_demand)
            weather_effect, weather_indicator = self.add_weather_effects(base_demand)
            
            # Stack all feature indicators
            features = torch.stack([
                promo_indicator,
                weather_indicator,
                torch.zeros(self.periods).refine_names('time'),  # sports placeholder
                torch.zeros(self.periods).refine_names('time'),  # school placeholder
                torch.zeros(self.periods).refine_names('time')   # holiday placeholder
            ], dim='features')
            
            # Combine all effects
            final_demand = base_demand + promo_effect + weather_effect
            
            # Add noise
            noise = torch.randn(self.periods) * (base_demand.rename(None).std() * 0.05)
            final_demand = torch.maximum(
                torch.zeros_like(final_demand.rename(None)),
                final_demand.rename(None) + noise
            ).refine_names('time')

        except Exception as e:
            raise ValueError(f"Error generating features: {str(e)}")

        # Create dates
        dates = pd.date_range(
            start=self.start_date,
            periods=self.periods,
            freq='D'
        )

        # Create temporal features
        temporal_features = self.create_temporal_features(pd.Series(dates))

        # Store feature dates for reference
        feature_dates = {
            'promotions': promo_indicator.rename(None).nonzero().squeeze().tolist(),
            'weather': weather_indicator.rename(None).nonzero().squeeze().tolist(),
            'sports': [],
            'school': [],
            'holidays': []
        }

        # Validate final shapes
        assert features.size('time') == self.periods
        assert features.size('features') == len(self.feature_names)
        assert temporal_features.size('time') == self.periods
        assert temporal_features.size('temporal_features') == len(self.temporal_feature_names)
        assert final_demand.size('time') == self.periods

        return DatasetFeatures(
            features=features,                    # [time, features]
            temporal=temporal_features,           # [time, temporal_features]
            target=final_demand,                 # [time]
            dates=pd.Series(dates),
            feature_dates=feature_dates
        )


if __name__ == "__main__":
    # Test data generation
    generator = SyntheticDataGenerator()
    data = generator.generate_data()
    
    print("\nFeature tensor shape:", dict(data.features.shape))
    print("Temporal features shape:", dict(data.temporal.shape))
    print("Target shape:", dict(data.target.shape))
    print("\nFeature names:", generator.feature_names)
    print("Temporal feature names:", generator.temporal_feature_names)
    print("\nFirst few dates:", data.dates.head())
    print("\nFeature dates:", data.feature_dates)