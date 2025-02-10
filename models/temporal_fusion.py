import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import time
from typing import Union, Tuple


class TimeDistributed(nn.Module):
    """Applies a module over multiple time steps"""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch, time, features]
        Returns:
            Output tensor of shape [batch, time, hidden_size]
        """
        batch_size, time_steps, features = x.size()
        # Reshape to [batch * time, features]
        x_reshaped = x.contiguous().view(-1, features)
        # Apply module
        y_reshaped = self.module(x_reshaped)
        # Reshape back to [batch, time, hidden]
        return y_reshaped.view(batch_size, time_steps, -1)


class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer model"""

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Save parameters
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Feature processing
        self.feature_layer = TimeDistributed(
            nn.Sequential(
                nn.Linear(num_features, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        )

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3,
        )

        # Output layer
        self.output_layer = TimeDistributed(
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch, time, features]
        Returns:
            Output tensor of shape [batch, time, 1]
        """
        # Process features
        x = self.feature_layer(x)

        # Apply transformer
        # No need for position embeddings as temporal info is in the features
        x = self.transformer(x)

        # Generate predictions
        return self.output_layer(x).squeeze(-1)


class TFTModel:
    """Wrapper class for training and inference"""

    def __init__(
        self,
        num_features: int,
        seq_length: int = 30,
        batch_size: int = 128,
        device: str = "cuda",
    ):
        self.model = TemporalFusionTransformer(
            num_features=num_features,
            hidden_size=256,
            num_heads=8,
        ).to(device)
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.device = device

    def create_sequences(
        self, X: torch.Tensor, y: torch.Tensor = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Create sequences for training or prediction

        Args:
            X: Input tensor with shape [time, features]
            y: Optional target tensor with shape [time]

        Returns:
            If y is None:
                X_seq: Tensor with shape [batch, seq_length, features]
            else:
                Tuple(
                    X_seq: Tensor with shape [batch, seq_length, features]
                    y_seq: Tensor with shape [batch]
                )
        """
        X_seqs = []
        y_seqs = []

        # Create sequences
        for i in range(len(X) - self.seq_length + 1):
            X_seqs.append(X[i : i + self.seq_length])
            if y is not None:
                y_seqs.append(y[i + self.seq_length - 1])

        # Stack sequences
        X_seq = torch.stack(X_seqs)

        if y is not None:
            y_seq = torch.stack(y_seqs)
            return X_seq, y_seq
        return X_seq

    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 100,
        early_stopping: bool = True,
    ):
        """Train the model

        Args:
            X: Input tensor with shape [time, features]
            y: Target tensor with shape [time]
            epochs: Number of training epochs
            early_stopping: Whether to use early stopping
        """
        # Validate input dimensions
        if X.size(1) != self.model.num_features:
            raise ValueError(
                f"Expected {self.model.num_features} features, got {X.size(1)}"
            )

        # Create sequences
        X_seq, y_seq = self.create_sequences(X, y)

        # Create dataset
        dataset = TensorDataset(X_seq, y_seq)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=0.001, weight_decay=0.01
        )
        scheduler = OneCycleLR(
            optimizer,
            max_lr=0.01,
            epochs=epochs,
            steps_per_epoch=len(loader),
            pct_start=0.3,
            anneal_strategy="cos",
        )

        # Training loop setup
        best_loss = float("inf")
        patience = 20
        patience_counter = 0
        training_start = time.time()
        last_log = time.time()

        for epoch in range(epochs):
            epoch_loss = 0
            self.model.train()

            # Log progress every minute
            if time.time() - last_log > 60:
                elapsed = time.time() - training_start
                print(
                    f"Epoch {epoch}/{epochs}, Time elapsed: {elapsed / 60:.1f}m"
                )
                if torch.cuda.is_available():
                    print(
                        f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.1f}GB"
                    )
                last_log = time.time()

            batch_count = 0
            for batch_X, batch_y in loader:
                # Move to device
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                predictions = outputs[:, -1]  # Take last timestep prediction

                loss = F.mse_loss(predictions, batch_y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )

                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                batch_count += 1

            epoch_loss /= batch_count

            # Early stopping check
            if early_stopping:
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Generate predictions

        Args:
            X: Input tensor with shape [time, features]

        Returns:
            Predictions tensor with shape [time]
        """
        # Validate input dimension
        if X.size(1) != self.model.num_features:
            raise ValueError(
                f"Expected {self.model.num_features} features, got {X.size(1)}"
            )

        # Create sequences
        X_seq = self.create_sequences(X)

        self.model.eval()
        with torch.no_grad():
            predictions = []
            # Process in batches
            for i in range(0, len(X_seq), self.batch_size):
                batch_X = X_seq[i : i + self.batch_size].to(self.device)
                batch_preds = self.model(batch_X)[:, -1]  # Take last timestep
                predictions.extend(batch_preds.cpu().numpy())

        # For the first seq_length-1 points, use the first prediction
        pad_predictions = [predictions[0]] * (self.seq_length - 1)
        final_predictions = np.concatenate([pad_predictions, predictions])

        return final_predictions
