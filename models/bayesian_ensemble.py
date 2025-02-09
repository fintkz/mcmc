import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from utils import peak_weighted_loss


class BayesianLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
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
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters following the way proposed in the paper."""
        nn.init.kaiming_normal_(self.weight_mu, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.weight_rho, -3)
        nn.init.constant_(self.bias_mu, 0.0)
        nn.init.constant_(self.bias_rho, -3)
        
    def forward(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        """Forward pass with proper tensor name handling"""
        # Remove names for operations
        x = x.rename(None)
        
        # Validate input dimension
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} input features, got {x.size(-1)}")

        # Hidden layers
        for layer in self.hidden_layers:
            x = layer(x, sample)  # BayesianLinear already handles unnamed tensors
            x = F.elu(x.rename(None))  # Explicitly remove names before ELU
            x = self.dropout(x)  # Dropout works on unnamed tensors
        
        # Output layer
        output = self.output_layer(x, sample)
        
        # Restore names and reshape to match expected dimensions
        return output.squeeze(-1).refine_names('batch')


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
        # Remove names for operations
        x = x.rename(None)
        
        # Validate input dimension
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} input features, got {x.size(-1)}")

        # Hidden layers
        for layer in self.hidden_layers:
            x = layer(x, sample)
            x = F.elu(x)  # ELU on unnamed tensor
            x = self.dropout(x)
        
        # Output layer
        output = self.output_layer(x, sample)
        
        # Restore names and reshape to match expected dimensions
        return output.squeeze(-1).refine_names('batch')


class GPUBayesianEnsemble:
    def __init__(self, input_dim: int, n_models: int = 5, device: str = None, 
                 batch_size: int = 32):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.input_dim = input_dim
        self.n_models = n_models
        self.batch_size = batch_size
        self.models = []

    def train(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 400, 
            num_samples: int = 10):
        """Train the ensemble with named tensors"""
        # Ensure inputs have proper names and are on CPU
        X = X.cpu().refine_names('time', 'features')
        y = y.cpu().refine_names('time')
        
        # Validate input dimensions
        if X.size('features') != self.input_dim:
            raise ValueError(
                f"Input dimension mismatch. Expected {self.input_dim}, got {X.size('features')}"
            )

        # Create dataset (TensorDataset doesn't support named tensors)
        dataset = TensorDataset(X.rename(None), y.rename(None))
        loader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            pin_memory=True,
            num_workers=0  # Avoid multiprocessing issues
        )

        # Clear any existing models
        self.models = []

        for i in range(self.n_models):
            model = BayesianNetwork(self.input_dim).to(self.device)
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=0.001, 
                weight_decay=0.01
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
                    # Move to GPU and restore names
                    batch_X = batch_X.to(self.device).refine_names('batch', 'features')
                    batch_y = batch_y.to(self.device).refine_names('batch')

                    optimizer.zero_grad()

                    # Multiple forward passes for MC Dropout
                    predictions = []
                    for _ in range(num_samples):
                        pred = model(batch_X, sample=True)
                        predictions.append(pred.rename(None))
                    
                    predictions = torch.stack(predictions, dim=0)  # [samples, batch]
                    pred_mean = predictions.mean(0)  # Average over samples

                    loss = peak_weighted_loss(pred_mean, batch_y.rename(None))

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
                    print(f"Model {i + 1}/{self.n_models}: Early stopping at epoch {epoch}")
                    break

                if (epoch + 1) % 20 == 0:
                    print(
                        f"Model {i + 1}/{self.n_models}, "
                        f"Epoch {epoch + 1}/{epochs}, "
                        f"Loss: {epoch_loss:.4f}"
                    )

            self.models.append(model)
            torch.cuda.empty_cache()

        print(f"Finished training {self.n_models} models")

    def predict(self, X: torch.Tensor) -> tuple:
        """Generate predictions with uncertainty estimates using named tensors"""
        # Ensure input has proper names
        X = X.refine_names('time', 'features')
        
        # Validate input dimension
        if X.size('features') != self.input_dim:
            raise ValueError(
                f"Input dimension mismatch. Expected {self.input_dim}, got {X.size('features')}"
            )

        predictions = []
        for model in self.models:
            model.eval()
            model_preds = []

            with torch.no_grad():
                # Process in batches
                for i in range(0, len(X), self.batch_size):
                    batch_X = X[i:i + self.batch_size].to(self.device)
                    batch_preds = []

                    # MC Dropout samples
                    for _ in range(50):
                        pred = model(batch_X, sample=True)
                        batch_preds.append(pred.rename(None))

                    # Stack along samples dimension
                    batch_preds = torch.stack(batch_preds, dim='samples')
                    model_preds.extend(batch_preds.mean('samples').cpu().numpy())

            predictions.append(np.array(model_preds))
            torch.cuda.empty_cache()

        # Stack predictions from all models
        predictions = np.stack(predictions)
        mean_pred = np.mean(predictions, axis=0).flatten()
        std_pred = np.std(predictions, axis=0).flatten()

        return mean_pred, std_pred