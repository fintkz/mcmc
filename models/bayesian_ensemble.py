import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from utils import peak_weighted_loss


class BayesianLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight parameters with named dimensions
        weight_mu = torch.zeros(out_features, in_features)
        weight_mu = weight_mu.refine_names('out_features', 'in_features')
        self.weight_mu = nn.Parameter(weight_mu)

        weight_rho = torch.zeros(out_features, in_features)
        weight_rho = weight_rho.refine_names('out_features', 'in_features')
        self.weight_rho = nn.Parameter(weight_rho)

        # Bias parameters with named dimensions
        bias_mu = torch.zeros(out_features)
        bias_mu = bias_mu.refine_names('out_features')
        self.bias_mu = nn.Parameter(bias_mu)

        bias_rho = torch.zeros(out_features)
        bias_rho = bias_rho.refine_names('out_features')
        self.bias_rho = nn.Parameter(bias_rho)

        # Initialize parameters
        self.weight_mu.data.normal_(0, 0.05)
        self.bias_mu.data.normal_(0, 0.05)
        self.weight_rho.data.fill_(-3)
        self.bias_rho.data.fill_(-3)

        self.prior_std = prior_std

    def forward(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        # Ensure input has proper names
        x = x.refine_names('batch', 'features')
        
        # Validate input dimension
        if x.size('features') != self.in_features:
            raise ValueError(f"Expected {self.in_features} input features, got {x.size('features')}")

        if sample:
            weight = (self.weight_mu + 
                     torch.log1p(torch.exp(self.weight_rho)) * 
                     torch.randn_like(self.weight_mu))
            bias = (self.bias_mu + 
                   torch.log1p(torch.exp(self.bias_rho)) * 
                   torch.randn_like(self.bias_mu))
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        # Matrix multiplication requires unnamed tensors
        output = torch.matmul(x.rename(None), weight.rename(None).t()) + bias.rename(None)
        # Restore names
        return output.refine_names('batch', 'out_features')


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
        # Ensure input has proper names
        x = x.refine_names('batch', 'features')
        
        # Validate input dimension
        if x.size('features') != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} input features, got {x.size('features')}")

        for layer in self.hidden_layers:
            x = F.elu(layer(x, sample))
            x = self.dropout(x.rename(None)).refine_names('batch', 'features')
        
        output = self.output_layer(x, sample)
        return output.align_to('batch', 'target')  # Rename out_features to target


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