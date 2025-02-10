from prophet import Prophet
import pandas as pd
import numpy as np
from typing import Optional, List, Union


class ProphetModel:
    """Prophet model wrapper for time series forecasting"""

    def __init__(self, changepoint_prior_scale: float = 0.05):
        """Initialize Prophet model

        Args:
            changepoint_prior_scale: Flexibility of the trend (0.05 by default)
        """
        self.model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
        )

    def train_and_predict(
        self,
        dates: pd.Series,
        y: np.ndarray,
        features: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Train model and generate predictions

        Args:
            dates: Series of dates
            y: Target values array of shape [time]
            features: Optional feature array of shape [time, n_features]
            feature_names: Optional list of feature names

        Returns:
            Array of predictions with shape [time]
        """
        # Validate inputs
        if len(dates) != len(y):
            raise ValueError(
                f"dates and y must have same length, got {len(dates)} and {len(y)}"
            )
        if features is not None:
            if len(features) != len(y):
                raise ValueError(
                    f"features and y must have same length, got {len(features)} and {len(y)}"
                )
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(features.shape[1])]
            if len(feature_names) != features.shape[1]:
                raise ValueError(
                    f"Number of feature names ({len(feature_names)}) must match number of features ({features.shape[1]})"
                )

        try:
            # Prepare training data
            df = pd.DataFrame({"ds": dates, "y": y})

            # Add features if provided
            if features is not None:
                for i, name in enumerate(feature_names):
                    df[name] = features[:, i]
                    self.model.add_regressor(name)

            # Fit model
            self.model.fit(df)

            # Generate predictions
            future = pd.DataFrame({"ds": dates})
            if features is not None:
                for i, name in enumerate(feature_names):
                    future[name] = features[:, i]

            forecast = self.model.predict(future)
            return forecast["yhat"].values

        except Exception as e:
            print(f"Error in Prophet model: {str(e)}")
            raise
