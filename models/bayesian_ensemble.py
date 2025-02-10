import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from utils import peak_weighted_loss
from typing import Tuple


class BayesianLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features  # Store input dimension
        self.out_features = out_features

        # Weight parameters
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.zeros(out_features, in_features))

        # Bias parameters
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.zeros(out_features))

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(
            self.weight_mu, mode="fan_in", nonlinearity="relu"
        )
        nn.init.constant_(self.weight_rho, -3)
        nn.init.constant_(self.bias_mu, 0.0)
        nn.init.constant_(self.bias_rho, -3)

    def forward(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        """Forward pass with reparameterization trick"""
        # Ensure tensor is unnamed
        x = x.rename(None)

        # Validate input dimension
        if (
            x.size(-1) != self.in_features
        ):  # Use in_features instead of input_dim
            raise ValueError(
                f"Expected {self.in_features} input features, got {x.size(-1)}"
            )

        if sample:
            weight = self.weight_mu + torch.randn_like(
                self.weight_mu
            ) * torch.exp(self.weight_rho)
            bias = self.bias_mu + torch.randn_like(self.bias_mu) * torch.exp(
                self.bias_rho
            )
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)  # Return unnamed tensor


class BayesianNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: list = [256, 128, 64]):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes

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
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        """Forward pass with proper tensor name handling"""
        # Store original names if any
        original_names = x.names

        # Remove names for operations
        x = x.rename(None)

        # Validate input dimension
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} input features, got {x.size(-1)}"
            )

        # Hidden layers
        for layer in self.hidden_layers:
            x = layer(
                x, sample
            )  # BayesianLinear already handles unnamed tensors
            x = F.elu(x)  # Already unnamed
            x = self.dropout(x)

        # Output layer
        output = self.output_layer(x, sample)

        # Restore appropriate names based on the output shape
        # The output will have shape [batch_size, 1] before squeezing
        output = output.squeeze(-1)

        # If the input had a batch dimension, name it accordingly
        if len(original_names) > 0 and "batch" in original_names:
            output = output.refine_names("batch")

        return output


class GPUBayesianEnsemble(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_models: int = 10,
        device: str = None,
        batch_size: int = 128,
    ):
        super().__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.input_dim = input_dim
        self.n_models = n_models
        self.batch_size = batch_size
        self.models = []

    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 400,
        num_samples: int = 10,
    ):
        """Train the ensemble with named tensors"""
        # Store original names
        X_names = X.names
        y_names = y.names

        # Remove names before creating dataset
        X = X.rename(None)
        y = y.rename(None)

        # Create dataset
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Clear existing models
        self.models = []

        try:
            for i in range(self.n_models):
                print(f"\nTraining model {i + 1}/{self.n_models}")
                model = BayesianNetwork(self.input_dim).to(self.device)

                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=1e-4, weight_decay=0.01
                )

                num_steps = len(loader) * epochs
                warmup_steps = num_steps // 10

                def lr_lambda(step):
                    if step < warmup_steps:
                        return float(step) / float(max(1, warmup_steps))
                    return 0.1 + 0.9 * (
                        1.0
                        - float(step - warmup_steps)
                        / float(max(1, num_steps - warmup_steps))
                    )

                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lr_lambda
                )

                best_loss = float("inf")
                patience = 20
                patience_counter = 0

                for epoch in range(epochs):
                    model.train()
                    epoch_loss = 0

                    for batch_X, batch_y in loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)

                        optimizer.zero_grad()

                        batch_losses = []
                        for _ in range(num_samples):
                            pred = model(batch_X, sample=True)
                            loss = peak_weighted_loss(pred, batch_y)
                            batch_losses.append(loss)

                        loss = torch.stack(batch_losses).mean()
                        loss.backward()

                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=1.0
                        )

                        optimizer.step()
                        scheduler.step()

                        epoch_loss += loss.item()

                    if (epoch + 1) % 20 == 0:
                        print(
                            f"Model {i + 1}/{self.n_models}, Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}"
                        )

                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        print(
                            f"Model {i + 1}/{self.n_models}: Early stopping at epoch {epoch}"
                        )
                        break

                self.models.append(model)
                torch.cuda.empty_cache()

            print(f"Finished training {self.n_models} models")

        finally:
            torch.cuda.empty_cache()

    def predict(
        self, X: torch.Tensor, num_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate predictions with uncertainty estimates"""
        # Store original names
        X_names = X.names

        # Remove names for operations
        X = X.rename(None)

        # Set all models to evaluation mode
        for model in self.models:
            model.eval()

        X = X.to(self.device)

        with torch.no_grad():
            all_preds = []

            for model in self.models:
                batch_preds = []

                for _ in range(num_samples):
                    pred = model(X, sample=True)
                    batch_preds.append(pred)

                model_preds = torch.stack(batch_preds, dim=0)
                all_preds.append(model_preds)

            ensemble_preds = torch.stack(all_preds, dim=0)

            all_predictions = ensemble_preds.reshape(
                -1, ensemble_preds.size(-1)
            )

            mean = all_predictions.mean(dim=0)
            std = all_predictions.std(dim=0)

            # Restore appropriate names
            if "batch" in X_names:
                mean = mean.refine_names("batch")
                std = std.refine_names("batch")

            return mean, std

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the ensemble"""
        return self.predict(X)
