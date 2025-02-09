import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from utils import peak_weighted_loss


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_std=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight parameters
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.zeros(out_features, in_features))

        # Bias parameters
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.zeros(out_features))

        # Initialize parameters
        self.weight_mu.data.normal_(0, 0.05)
        self.bias_mu.data.normal_(0, 0.05)
        self.weight_rho.data.fill_(-3)
        self.bias_rho.data.fill_(-3)

        self.prior_std = prior_std

    def forward(self, x, sample=False):
        if sample:
            weight = self.weight_mu + torch.log1p(
                torch.exp(self.weight_rho)
            ) * torch.randn_like(self.weight_mu)
            bias = self.bias_mu + torch.log1p(
                torch.exp(self.bias_rho)
            ) * torch.randn_like(self.bias_mu)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)


class BayesianNetwork(nn.Module):
    def __init__(self, input_dim, hidden_sizes=[256, 128, 64]):
        super().__init__()

        self.hidden_layers = nn.ModuleList()

        # Input layer
        self.hidden_layers.append(BayesianLinear(input_dim, hidden_sizes[0]))

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(
                BayesianLinear(hidden_sizes[i], hidden_sizes[i + 1])
            )

        # Output layer
        self.output_layer = BayesianLinear(hidden_sizes[-1], 1)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, sample=False):
        for layer in self.hidden_layers:
            x = F.elu(layer(x, sample))
            x = self.dropout(x)
        return self.output_layer(x, sample)


class GPUBayesianEnsemble:
    def __init__(self, input_dim, n_models=5, device=None, batch_size=32):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.input_dim = input_dim
        self.n_models = n_models
        self.models = []
        self.batch_size = batch_size

    def create_temporal_features(self, X):
        """Create temporal features for the input data"""
        batch_size = X.shape[0]

        # Create time index features
        time_index = np.arange(batch_size).reshape(-1, 1) / batch_size

        # Create seasonal features
        day_sin = np.sin(2 * np.pi * time_index * 365)
        day_cos = np.cos(2 * np.pi * time_index * 365)
        week_sin = np.sin(2 * np.pi * time_index * 52)
        week_cos = np.cos(2 * np.pi * time_index * 52)

        # Create lagged features
        lagged_features = []
        target_column = X[:, -1:] if len(X.shape) > 1 else X.reshape(-1, 1)

        for lag in [1, 7, 14, 30]:
            lagged = np.zeros((batch_size, 1))
            lagged[lag:] = target_column[:-lag]
            lagged_features.append(lagged)

    def train(self, X, y, epochs=800, num_samples=10):
        """Train the ensemble of Bayesian neural networks"""
        # Add temporal features
        X = self.create_temporal_features(X)

        # Calculate actual input dimension after adding features
        actual_input_dim = X.shape[1]

        # Keep data on CPU initially
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)

        if len(y.shape) == 1:
            y = y.view(-1, 1)

        dataset = TensorDataset(X, y)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, pin_memory=False
        )

        for i in range(self.n_models):
            model = BayesianNetwork(actual_input_dim).to(self.device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=0.001, weight_decay=0.01
            )

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=0.01,
                epochs=epochs,
                steps_per_epoch=len(loader),
                pct_start=0.3,
                anneal_strategy="cos",
            )

            best_loss = float("inf")
            patience = 25
            patience_counter = 0

            for epoch in range(epochs):
                epoch_loss = 0
                model.train()

                for batch_X, batch_y in loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    optimizer.zero_grad()

                    # Multiple forward passes for MC Dropout
                    predictions = torch.stack(
                        [model(batch_X, sample=True) for _ in range(num_samples)]
                    )
                    pred_mean = predictions.mean(0)

                    loss = peak_weighted_loss(pred_mean, batch_y)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()

                    epoch_loss += loss.item()

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

                if (epoch + 1) % 20 == 0:
                    print(
                        f"Model {i + 1}/{self.n_models}, Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}"
                    )

            self.models.append(model)
            # Clear some memory
            torch.cuda.empty_cache()

    def predict(self, X):
        """Generate predictions with uncertainty estimates"""
        # Add temporal features for prediction
        X = self.create_temporal_features(X)

        # Keep data on CPU initially
        X = torch.FloatTensor(X)
        predictions = []

        for model in self.models:
            model.eval()
            model_preds = []

            with torch.no_grad():
                # Process in batches to avoid memory issues
                for i in range(0, len(X), self.batch_size):
                    batch_X = X[i : i + self.batch_size].to(self.device)
                    batch_preds = []

                    for _ in range(50):  # MC Dropout samples
                        pred = model(batch_X, sample=True)
                        batch_preds.append(pred.cpu().numpy())

                    model_preds.extend(np.mean(batch_preds, axis=0))

            predictions.append(np.array(model_preds))
            # Clear some memory
            torch.cuda.empty_cache()

        predictions = np.stack(predictions)
        mean_pred = np.mean(predictions, axis=0).flatten()
        std_pred = np.std(predictions, axis=0).flatten()

        return mean_pred, std_pred
