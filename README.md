
# MCMC Grocery Demand Simulation

A simple, fun project to simulate and preprocess grocery store demand data using Python!

I started this project to explore time series data generation and preprocessing techniques. If you have any suggestions or find areas that could be improved, feel free to open an issue or contribute with a pull request. I'm here to learn and grow, and I appreciate any feedback or contributions.

## Installation

Requirements:

- Python 3.8+
- NumPy
- Pandas
- scikit-learn

To get started, clone the repository and install the required packages:

```sh
git clone https://github.com/yourusername/mcmc.git
cd mcmc
pip install -r requirements.txt
```

## Usage

To generate synthetic grocery store demand data and preprocess it, you can use the provided functions in `data.py`.

### Generating Synthetic Data

The `generate_synthetic_grocery_data` function generates synthetic grocery store demand data with yearly and weekly seasonality, random noise, and spikes.

Example usage:

```python
from data import generate_synthetic_grocery_data

df = generate_synthetic_grocery_data(n_days=365)
print(df.head())
```

### Preprocessing Data

The `preprocess_data` function preprocesses the generated data by adding temporal features, lag features, rolling statistics, and scaling the features.

Example usage:

```python
from data import preprocess_data

X_scaled, y_scaled, scaler_y, feature_columns = preprocess_data(df)
print(X_scaled[:5])
print(y_scaled[:5])
```

## Configuration

The main functions you'll care about are:

### Generate Synthetic Data

```python
generate_synthetic_grocery_data(n_days=365)
```

Generates synthetic grocery store demand data for the specified number of days.

### Preprocess Data

```python
preprocess_data(df)
```

Preprocesses the data with temporal and statistical features, and scales the features and target.

## Example

Here's a complete example of generating and preprocessing the data:

```python
from data import generate_synthetic_grocery_data, preprocess_data

# Generate synthetic data
df = generate_synthetic_grocery_data(n_days=365)

# Preprocess the data
X_scaled, y_scaled, scaler_y, feature_columns = preprocess_data(df)

# Print the first few rows of the scaled features and target
print(X_scaled[:5])
print(y_scaled[:5])
```

## Contributing

Contributions are always welcome!

Just open a PR and I'll review it!
