from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List


class ProphetModel:
    def __init__(self, seasonality_mode="multiplicative"):
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode=seasonality_mode,
        )
        self.has_external = False

    def train_and_predict(self, dates: pd.DatetimeIndex, y: np.ndarray) -> np.ndarray:
        """Train the model and generate predictions"""

        # Should validate inputs and handle potential errors
        if len(dates) != len(y):
            raise ValueError("dates and y must have the same length")
        
        # Should handle NaN/Inf values that could break Prophet
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            raise ValueError("Input contains NaN or Inf values")
        # Create DataFrame in Prophet format
        df = pd.DataFrame({
            'ds': dates,
            'y': y
        })
        
        # Fit the model
        self.model.fit(df)
        
        # Make predictions
        future = pd.DataFrame({'ds': dates})
        forecast = self.model.predict(future)
        
        return forecast['yhat'].values

    def add_external_features(self, feature_names: List[str]) -> None:
        """Add external regressors to Prophet"""
        for feature in feature_names:
            if feature in ["ds", "y"]:
                raise ValueError(f"Cannot use reserved name {feature} as external feature")
            self.model.add_regressor(feature)
        self.has_external = bool(feature_names)  # More accurate than just True

    def fit(self, df):
        """
        Fit Prophet model
        df must contain 'ds' (dates) and 'y' (target) columns
        """
        self.model.fit(df)

    def predict(self, df):
        """Make predictions"""
        forecast = self.model.predict(df)
        return forecast

    def plot_components(self, forecast):
        """Plot Prophet's decomposition"""
        self.model.plot_components(forecast)
        plt.tight_layout()

    def plot_prediction(self, forecast, actual=None):
        """Plot predictions vs actuals"""
        fig = plt.figure(figsize=(15, 6))
        plt.plot(forecast["ds"], forecast["yhat"], label="Prediction", color="blue")
        plt.fill_between(
            forecast["ds"],
            forecast["yhat_lower"],
            forecast["yhat_upper"],
            color="blue",
            alpha=0.2,
            label="Uncertainty",
        )
        if actual is not None:
            plt.plot(actual["ds"], actual["y"], label="Actual", color="red", alpha=0.5)
        plt.title("Prophet Forecast with Uncertainty Intervals")
        plt.legend()
        plt.grid(True, alpha=0.3)
        return fig