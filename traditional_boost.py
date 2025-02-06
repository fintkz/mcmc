import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def train_and_evaluate_gb(X_train, y_train, X_test, y_test):
    """Train and evaluate traditional gradient boosting."""
    
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3
    )
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    
    return model, y_pred, mse

if __name__ == "__main__":
    from synthetic_data import generate_synthetic_grocery_data
    
    # Generate data
    df = generate_synthetic_grocery_data()
    X, y = preprocess_data(df)
    
    # Train test split
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train and evaluate
    model, y_pred, mse = train_and_evaluate_gb(X_train, y_train, X_test, y_test)
    print(f"Traditional Gradient Boosting MSE: {mse}")
