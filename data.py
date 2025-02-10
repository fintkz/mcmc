import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class DatasetFeatures:
    """Container for dataset features

    Attributes:
        features: Tensor with shape [time, features]
        temporal: Tensor with shape [time, temporal_features]
        target: Tensor with shape [time]
        dates: Pandas Series of dates
        feature_dates: Dict mapping feature names to lists of dates
    """

    features: torch.Tensor
    temporal: torch.Tensor
    target: torch.Tensor
    dates: pd.Series
    feature_dates: Dict


class SyntheticDataGenerator:
    def __init__(self, start_date: str = "2023-01-01", periods: int = 365):
        """Initialize data generator with parameters"""
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.periods = periods
        self.feature_names = [
            "promotions_active",
            "weather_event",
            "sports_event",
            "school_term",
            "holiday",
        ]
        self.temporal_feature_names = [
            "day_sin",
            "day_cos",
            "week_sin",
            "week_cos",
            "month_sin",
            "month_cos",
        ]

    def generate_base_demand(self) -> torch.Tensor:
        """Generate base demand with trend and seasonality"""
        time = torch.arange(self.periods, dtype=torch.float32)

        # Trend
        trend = 1000 + time * 0.5

        # Yearly seasonality
        yearly = 200 * torch.sin(2 * np.pi * time / 365)

        # Weekly seasonality
        weekly = 50 * torch.sin(2 * np.pi * time / 7)

        base_demand = trend + yearly + weekly
        return base_demand

    def add_promotions(
        self, base_demand: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add promotion effects with named tensors"""
        # Generate random promotion dates (about 2 per month)
        promo_dates = torch.sort(torch.randperm(self.periods)[:24])[0]

        # Effect lasts for 3-5 days with peak on second day
        promo_effect = torch.zeros(self.periods)
        promo_indicator = torch.zeros(self.periods)

        for date in promo_dates:
            duration = torch.randint(3, 6, (1,)).item()
            effect_pattern = torch.tensor([0.5, 1.0, 0.7, 0.3, 0.1][:duration])
            end_idx = min(date.item() + duration, self.periods)
            promo_effect[date:end_idx] += effect_pattern[: end_idx - date]
            promo_indicator[date] = 1

        return promo_effect * 300, promo_indicator

    def add_weather_effects(
        self, base_demand: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add weather effects with named tensors"""
        weather_dates = torch.sort(torch.randperm(self.periods)[:30])[0]

        weather_effect = torch.zeros(self.periods)
        weather_indicator = torch.zeros(self.periods)

        for date in weather_dates:
            duration = torch.randint(1, 4, (1,)).item()
            effect = torch.randn(1).item() * 0.2  # Random effect between -0.2 and 0.2
            end_idx = min(date.item() + duration, self.periods)
            weather_effect[date:end_idx] = effect
            weather_indicator[date] = 1

        return base_demand * weather_effect, weather_indicator

    def add_sports_events(
        self, base_demand: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add sports event effects with named tensors"""
        # Generate random sports event dates (about 2 per month)
        sports_dates = torch.sort(torch.randperm(self.periods)[:24])[0]

        sports_effect = torch.zeros(self.periods)
        sports_indicator = torch.zeros(self.periods)

        for date in sports_dates:
            effect = torch.randn(1).item() * 0.3 + 0.2  # Random positive effect
            sports_effect[date] = effect
            sports_indicator[date] = 1

        return base_demand * sports_effect, sports_indicator

    def add_school_terms(
        self, base_demand: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add school term effects with named tensors"""
        # Create term dates (roughly following typical school calendar)
        school_indicator = torch.zeros(self.periods)
        school_effect = torch.zeros(self.periods)

        # Term 1: Jan-Mar (days 0-90)
        school_indicator[0:90] = 1
        # Term 2: Apr-Jun (days 91-181)
        school_indicator[105:181] = 1
        # Term 3: Jul-Sep (days 182-273)
        school_indicator[196:273] = 1
        # Term 4: Oct-Dec (days 274-365)
        school_indicator[288:365] = 1

        # Add small random effect during school terms
        school_effect = school_indicator * (torch.randn(self.periods) * 0.1 + 0.15)

        return base_demand * school_effect, school_indicator

    def add_holidays(
        self, base_demand: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add holiday effects with named tensors"""
        holiday_dates = [
            0,  # New Year's Day
            45,  # Valentine's Day
            78,  # Easter
            120,  # Mother's Day
            150,  # Father's Day
            185,  # Independence Day
            243,  # Labor Day
            303,  # Halloween
            329,  # Thanksgiving
            359,  # Christmas
        ]

        holiday_effect = torch.zeros(self.periods)
        holiday_indicator = torch.zeros(self.periods)

        for date in holiday_dates:
            if date < self.periods:
                effect = torch.randn(1).item() * 0.2 + 0.4  # Random positive effect
                holiday_effect[date] = effect
                holiday_indicator[date] = 1

                # Add pre-holiday effect
                if date > 0:
                    holiday_effect[date - 1] = effect * 0.5

        return base_demand * holiday_effect, holiday_indicator

    def create_temporal_features(self, dates: pd.Series) -> torch.Tensor:
        """Create temporal features with named dimensions"""
        time = torch.arange(self.periods, dtype=torch.float32)

        # Create cyclical features
        day_of_year = torch.tensor(dates.dt.dayofyear.values, dtype=torch.float32)
        week_of_year = torch.tensor(
            dates.dt.isocalendar().week.values, dtype=torch.float32
        )
        month = torch.tensor(dates.dt.month.values, dtype=torch.float32)

        # Stack all temporal features
        temporal_features = torch.stack(
            [
                torch.sin(2 * np.pi * day_of_year / 365),
                torch.cos(2 * np.pi * day_of_year / 365),
                torch.sin(2 * np.pi * week_of_year / 52),
                torch.cos(2 * np.pi * week_of_year / 52),
                torch.sin(2 * np.pi * month / 12),
                torch.cos(2 * np.pi * month / 12),
            ],
            dim=1,
        )

        return temporal_features

    def generate_data(self) -> DatasetFeatures:
        """Generate complete synthetic dataset"""
        if self.periods < 365:
            raise ValueError(f"Periods must be >= 365, got {self.periods}")

        try:
            # Generate base demand
            base_demand = self.generate_base_demand()

            # Add various effects
            promo_effect, promo_indicator = self.add_promotions(base_demand)
            weather_effect, weather_indicator = self.add_weather_effects(base_demand)
            sports_effect, sports_indicator = self.add_sports_events(base_demand)
            school_effect, school_indicator = self.add_school_terms(base_demand)
            holiday_effect, holiday_indicator = self.add_holidays(base_demand)

            # Stack all feature indicators
            features = torch.stack(
                [
                    promo_indicator,
                    weather_indicator,
                    sports_indicator,
                    school_indicator,
                    holiday_indicator,
                ],
                dim=1,
            )

            # Combine all effects
            final_demand = (
                base_demand
                + promo_effect
                + weather_effect
                + sports_effect
                + school_effect
                + holiday_effect
            )

            # Add noise
            noise = torch.randn(self.periods) * (base_demand.std() * 0.05)
            final_demand = torch.maximum(
                torch.zeros_like(final_demand),
                (final_demand + noise),
            )

            # Create dates
            dates = pd.date_range(start=self.start_date, periods=self.periods, freq="D")
            dates_series = pd.Series(dates)

            # Generate temporal features
            temporal_features = self.create_temporal_features(dates_series)

            # Store feature dates for reference
            feature_dates = {
                "promotions": promo_indicator.nonzero().squeeze().tolist(),
                "weather": weather_indicator.nonzero().squeeze().tolist(),
                "sports": sports_indicator.nonzero().squeeze().tolist(),
                "school": school_indicator.nonzero().squeeze().tolist(),
                "holidays": holiday_indicator.nonzero().squeeze().tolist(),
            }

            # Validate final shapes
            assert features.size(0) == self.periods
            assert features.size(1) == len(self.feature_names)
            assert temporal_features.size(0) == self.periods
            assert temporal_features.size(1) == len(self.temporal_feature_names)
            assert final_demand.size(0) == self.periods

            return DatasetFeatures(
                features=features,
                temporal=temporal_features,
                target=final_demand,
                dates=dates_series,
                feature_dates=feature_dates,
            )

        except Exception as e:
            raise ValueError(f"Error generating features: {str(e)}")


if __name__ == "__main__":
    # Test data generation
    generator = SyntheticDataGenerator()
    data = generator.generate_data()

    print("\nFeature tensor shape:", data.features.shape)
    print("Temporal features shape:", data.temporal.shape)
    print("Target shape:", data.target.shape)
    print("\nFeature names:", generator.feature_names)
    print("Temporal feature names:", generator.temporal_feature_names)
    print("\nFirst few dates:", data.dates.head())
    print("\nFeature dates:", data.feature_dates)
