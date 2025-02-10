import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import calculate_mape
import numpy as np
from typing import Tuple, List


class BayesianNetwork(nn.Module):
    """Bayesian Neural Network with Gaussian variational inference"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_hidden: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden

        # Initialize layers
        layers = []
        # Input layer
        layers.extend(
            [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        )
        # Hidden layers
        for _ in range(n_hidden - 1):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
        # Output layer - mean and log variance
        self.network = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.logvar_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network

        Args:
            x: Input tensor of shape [batch_size, input_dim]

        Returns:
            Tuple of:
                mean: Predicted mean of shape [batch_size, 1]
                logvar: Predicted log variance of shape [batch_size, 1]
        """
        features = self.network(x)
        mean = self.mean_head(features)
        logvar = self.logvar_head(features)
        return mean, logvar

    def sample_elbo(
        self, x: torch.Tensor, y: torch.Tensor, n_samples: int = 1
    ) -> torch.Tensor:
        """Compute ELBO loss

        Args:
            x: Input tensor of shape [batch_size, input_dim]
            y: Target tensor of shape [batch_size]
            n_samples: Number of MC samples

        Returns:
            ELBO loss value
        """
        mean, logvar = self(x)
        var = torch.exp(logvar)

        # Compute KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - var)

        # Sample and compute log likelihood
        dist = Normal(mean.squeeze(), var.sqrt().squeeze())
        log_likely = dist.log_prob(y).mean()

        return log_likely - kl_loss


class GPUBayesianEnsemble:
    """Ensemble of Bayesian Neural Networks"""

    def __init__(
        self,
        input_dim: int,
        n_networks: int = 5,
        hidden_dim: int = 64,
        n_hidden: int = 2,
        device: str = "cuda",
        rank: int = 0,
    ):
        self.device = device
        self.input_dim = input_dim
        self.n_networks = n_networks
        self.rank = rank

        # Create ensemble of networks
        self.networks = [
            BayesianNetwork(
                input_dim=input_dim, hidden_dim=hidden_dim, n_hidden=n_hidden
            ).to(device)
            for _ in range(n_networks)
        ]

        # Wrap networks in DDP if using distributed training
        if device == "cuda":
            self.networks = [
                DDP(net, device_ids=[rank]) for net in self.networks
            ]

        # Initialize optimizers
        self.optimizers = [
            torch.optim.Adam(net.parameters(), lr=0.001)
            for net in self.networks
        ]

    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 64,
        mc_samples: int = 10,
    ):
        """Train the ensemble

        Args:
            X: Input tensor of shape [time, features]
            y: Target tensor of shape [time]
            epochs: Number of training epochs
            batch_size: Batch size for training
            mc_samples: Number of Monte Carlo samples
        """
        # Validate inputs
        if X.size(1) != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, got {X.size(1)}"
            )
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have same length, got {len(X)} and {len(y)}"
            )

        try:
            # Create dataset and distributed sampler
            dataset = torch.utils.data.TensorDataset(X, y)
            sampler = (
                torch.utils.data.distributed.DistributedSampler(dataset)
                if dist.is_initialized()
                else None
            )
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(sampler is None),
                sampler=sampler,
            )

            for epoch in range(epochs):
                if sampler is not None:
                    sampler.set_epoch(epoch)

                epoch_loss = 0.0
                batch_count = 0

                for batch_x, batch_y in loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    # Train each network
                    for net, opt in zip(self.networks, self.optimizers):
                        opt.zero_grad()

                        # Generate MC samples
                        all_preds = []
                        for _ in range(mc_samples):
                            mean, logvar = net(batch_x)
                            var = torch.exp(logvar)
                            pred = Normal(
                                mean.squeeze(), var.sqrt().squeeze()
                            ).sample()
                            all_preds.append(pred)

                        # Average predictions
                        pred = torch.stack(all_preds).mean(0)

                        # Calculate MAPE loss
                        loss = calculate_mape(pred.cpu(), batch_y.cpu())
                        loss = torch.tensor(loss, device=self.device)

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                        opt.step()

                        epoch_loss += loss.item()
                        batch_count += 1

                if (epoch + 1) % 10 == 0 and self.rank == 0:
                    avg_loss = epoch_loss / batch_count
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

                # Check for NaN loss
                if torch.isnan(torch.tensor(epoch_loss)):
                    raise ValueError("Training loss became NaN")

        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise

    def predict(
        self, X: torch.Tensor, n_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate predictions with uncertainty

        Args:
            X: Input tensor of shape [time, features]
            n_samples: Number of MC samples for prediction

        Returns:
            Tuple of:
                mean: Mean predictions of shape [time]
                std: Standard deviation of predictions of shape [time]
        """
        all_preds = []

        # Get predictions from each network
        for net in self.networks:
            net.eval()
            with torch.no_grad():
                batch_preds = []
                for _ in range(n_samples):
                    mean, logvar = net(X)
                    var = torch.exp(logvar)
                    pred = Normal(mean.squeeze(), var.sqrt().squeeze()).sample()
                    batch_preds.append(pred)
                network_preds = torch.stack(batch_preds)
                all_preds.append(network_preds)

        # Stack predictions from all networks
        all_preds = torch.stack(all_preds)  # [n_networks, n_samples, time]

        # Compute mean and standard deviation
        mean_preds = all_preds.mean(dim=(0, 1))  # [time]
        std_preds = all_preds.std(dim=(0, 1))  # [time]

        return mean_preds.cpu(), std_preds.cpu()
