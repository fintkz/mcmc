import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple, NamedTuple
from dataclasses import dataclass


@dataclass
class DatasetFeatures:
    """Container for dataset features"""

    features: torch.Tensor  # Shape: [time, features]
    temporal: torch.Tensor  # Shape: [time, temporal_features]
    target: torch.Tensor  # Shape: [time]
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
        time = time.refine_names("time")

        # Trend
        trend = 1000 + time.rename(None) * 0.5

        # Yearly seasonality
        yearly = 200 * torch.sin(2 * np.pi * time.rename(None) / 365)

        # Weekly seasonality
        weekly = 50 * torch.sin(2 * np.pi * time.rename(None) / 7)

        base_demand = (trend + yearly + weekly).refine_names("time")
        return base_demand

    def add_promotions(
        self, base_demand: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add promotion effects with named tensors"""
        # Generate random promotion dates (about 2 per month)
        promo_dates = torch.sort(torch.randperm(self.periods)[:24])[0]

        # Effect lasts for 3-5 days with peak on second day
        promo_effect = torch.zeros(self.periods).refine_names("time")
        promo_indicator = torch.zeros(self.periods).refine_names("time")

        for date in promo_dates:
            duration = torch.randint(3, 6, (1,)).item()
            effect_pattern = torch.tensor([0.5, 1.0, 0.7, 0.3, 0.1][:duration])
            end_idx = min(date.item() + duration, self.periods)
            promo_effect.rename(None)[date:end_idx] += effect_pattern[
                : end_idx - date
            ]
            promo_indicator.rename(None)[date] = 1

        return promo_effect * 300, promo_indicator

    def add_weather_effects(
        self, base_demand: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add weather effects with named tensors"""
        weather_dates = torch.sort(torch.randperm(self.periods)[:30])[0]

        weather_effect = torch.zeros(self.periods).refine_names("time")
        weather_indicator = torch.zeros(self.periods).refine_names("time")

        for date in weather_dates:
            duration = torch.randint(1, 4, (1,)).item()
            effect = (
                torch.randn(1).item() * 0.2
            )  # Random effect between -0.2 and 0.2
            end_idx = min(date.item() + duration, self.periods)
            weather_effect.rename(None)[date:end_idx] = effect
            weather_indicator.rename(None)[date] = 1

        return base_demand * weather_effect, weather_indicator

    def add_sports_events(
        self, base_demand: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add sports event effects with named tensors"""
        # Generate random sports event dates (about 2 per month)
        sports_dates = torch.sort(torch.randperm(self.periods)[:24])[0]

        sports_effect = torch.zeros(self.periods).refine_names("time")
        sports_indicator = torch.zeros(self.periods).refine_names("time")

        for date in sports_dates:
            effect = torch.randn(1).item() * 0.3 + 0.2  # Random positive effect
            sports_effect.rename(None)[date] = effect
            sports_indicator.rename(None)[date] = 1

        return base_demand * sports_effect, sports_indicator

    def add_school_terms(
        self, base_demand: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add school term effects with named tensors"""
        # Create term dates (roughly following typical school calendar)
        school_indicator = torch.zeros(self.periods).refine_names("time")
        school_effect = torch.zeros(self.periods).refine_names("time")

        # Term 1: Jan-Mar (days 0-90)
        school_indicator.rename(None)[0:90] = 1
        # Term 2: Apr-Jun (days 91-181)
        school_indicator.rename(None)[105:181] = 1
        # Term 3: Jul-Sep (days 182-273)
        school_indicator.rename(None)[196:273] = 1
        # Term 4: Oct-Dec (days 274-365)
        school_indicator.rename(None)[288:365] = 1

        # Add small random effect during school terms
        school_effect = school_indicator.rename(None) * (
            torch.randn(self.periods) * 0.1 + 0.15
        )
        school_effect = school_effect.refine_names("time")

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

        holiday_effect = torch.zeros(self.periods).refine_names("time")
        holiday_indicator = torch.zeros(self.periods).refine_names("time")

        for date in holiday_dates:
            if date < self.periods:
                effect = (
                    torch.randn(1).item() * 0.2 + 0.4
                )  # Random positive effect
                holiday_effect.rename(None)[date] = effect
                holiday_indicator.rename(None)[date] = 1

                # Add pre-holiday effect
                if date > 0:
                    holiday_effect.rename(None)[date - 1] = effect * 0.5

        return base_demand * holiday_effect, holiday_indicator

    def create_temporal_features(self, dates: pd.Series) -> torch.Tensor:
        """Create temporal features with named dimensions"""
        time = torch.arange(self.periods, dtype=torch.float32)
        time = time.refine_names("time")

        # Create cyclical features
        day_of_year = torch.tensor(
            dates.dt.dayofyear.values, dtype=torch.float32
        )
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
            dim="temporal_features",
        )

        return temporal_features.align_to("time", "temporal_features")

    def generate_data(self) -> DatasetFeatures:
        """Generate complete synthetic dataset with named tensors"""
        if self.periods < 365:
            raise ValueError(
                "Periods should be at least 365 for meaningful seasonal patterns"
            )

        try:
            # Generate base demand
            base_demand = self.generate_base_demand()

            # Generate all effects and their indicators
            promo_effect, promo_indicator = self.add_promotions(base_demand)
            weather_effect, weather_indicator = self.add_weather_effects(
                base_demand
            )
            sports_effect, sports_indicator = self.add_sports_events(
                base_demand
            )
            school_effect, school_indicator = self.add_school_terms(base_demand)
            holiday_effect, holiday_indicator = self.add_holidays(base_demand)

            # Stack all feature indicators
            features = torch.stack(
                [
                    promo_indicator.rename(None),
                    weather_indicator.rename(None),
                    sports_indicator.rename(None),
                    school_indicator.rename(None),
                    holiday_indicator.rename(None),
                ],
                dim=1,
            )

            # Now refine the names
            features = features.refine_names("time", "features")

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
            noise = torch.randn(self.periods) * (
                base_demand.rename(None).std() * 0.05
            )
            final_demand = torch.maximum(
                torch.zeros_like(final_demand.rename(None)),
                (final_demand.rename(None) + noise),
            ).refine_names("time")

            # Create dates
            dates = pd.date_range(
                start=self.start_date, periods=self.periods, freq="D"
            )

            # Convert to Series for datetime accessors
            dates_series = pd.Series(dates)
            # Create temporal features (similar fix needed here)
            temporal_features = torch.stack(
                [
                    # Day of year features
                    torch.sin(
                        2
                        * np.pi
                        * torch.tensor(
                            dates_series.dt.dayofyear.values,
                            dtype=torch.float32,
                        )
                        / 365
                    ),
                    torch.cos(
                        2
                        * np.pi
                        * torch.tensor(
                            dates_series.dt.dayofyear.values,
                            dtype=torch.float32,
                        )
                        / 365
                    ),
                    # Week features
                    torch.sin(
                        2
                        * np.pi
                        * torch.tensor(
                            dates_series.dt.isocalendar().week.values,
                            dtype=torch.float32,
                        )
                        / 52
                    ),
                    torch.cos(
                        2
                        * np.pi
                        * torch.tensor(
                            dates_series.dt.isocalendar().week.values,
                            dtype=torch.float32,
                        )
                        / 52
                    ),
                    # Month features
                    torch.sin(
                        2
                        * np.pi
                        * torch.tensor(
                            dates_series.dt.month.values, dtype=torch.float32
                        )
                        / 12
                    ),
                    torch.cos(
                        2
                        * np.pi
                        * torch.tensor(
                            dates_series.dt.month.values, dtype=torch.float32
                        )
                        / 12
                    ),
                ],
                dim=1,
            )  # Use dim=1 for temporal_features dimension

            # Now refine the names
            temporal_features = temporal_features.refine_names(
                "time", "temporal_features"
            )

            # Store feature dates for reference
            feature_dates = {
                "promotions": promo_indicator.rename(None)
                .nonzero()
                .squeeze()
                .tolist(),
                "weather": weather_indicator.rename(None)
                .nonzero()
                .squeeze()
                .tolist(),
                "sports": sports_indicator.rename(None)
                .nonzero()
                .squeeze()
                .tolist(),
                "school": school_indicator.rename(None)
                .nonzero()
                .squeeze()
                .tolist(),
                "holidays": holiday_indicator.rename(None)
                .nonzero()
                .squeeze()
                .tolist(),
            }

            # Validate final shapes
            assert features.size("time") == self.periods
            assert features.size("features") == len(self.feature_names)
            assert temporal_features.size("time") == self.periods
            assert temporal_features.size("temporal_features") == len(
                self.temporal_feature_names
            )
            assert final_demand.size("time") == self.periods

            return DatasetFeatures(
                features=features,  # [time, features]
                temporal=temporal_features,  # [time, temporal_features]
                target=final_demand,  # [time]
                dates=dates_series,
                feature_dates=feature_dates,
            )

        except Exception as e:
            raise ValueError(f"Error generating features: {str(e)}")


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
