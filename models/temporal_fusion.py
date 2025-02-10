import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import TensorDataset, DataLoader
from utils import peak_weighted_loss
import numpy as np
from typing import Union
import time


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply module preserving batch and time dimensions"""
        # Input shape: [batch, time, features] or [batch, features]
        if len(x.shape) == 3:
            # Has time dimension
            batch_size, time_steps, n_features = x.shape
            # Reshape to [batch * time, features]
            x_reshape = x.reshape(-1, n_features)
            y = self.module(x_reshape)
            # Reshape back to [batch, time, output_features]
            return y.reshape(batch_size, time_steps, -1)
        else:
            # No time dimension, just apply module
            return self.module(x)


class GatedResidualNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # Main layers
        self.fc1 = TimeDistributed(nn.Linear(input_size, hidden_size))
        self.elu1 = nn.ELU()
        self.fc2 = TimeDistributed(nn.Linear(hidden_size, output_size))
        self.elu2 = nn.ELU()

        # Skip connection if input and output dimensions differ
        self.skip = (
            TimeDistributed(nn.Linear(input_size, output_size))
            if input_size != output_size
            else None
        )

        # Gating mechanism - FIXED: input size should be input_size + output_size
        self.gate = TimeDistributed(
            nn.Linear(input_size + output_size, output_size)
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Input shape: [batch, time, features]

        # Validate input dimension
        if x.size(-1) != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} features, got {x.size(-1)}"
            )

        # Main branch
        h = self.fc1(x)
        h = self.elu1(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.elu2(h)

        # Skip connection
        if self.skip is not None:
            x = self.skip(x)

        # Gating mechanism
        combined = torch.cat([x, h], dim=-1)
        gate = self.gate(combined)
        gate = torch.sigmoid(gate)

        # Element-wise multiplication and layer norm
        weighted = gate * h + (1 - gate) * x
        return self.layer_norm(weighted)


class TemporalFusionTransformer(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_size: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size

        # Project input to hidden size
        self.input_projection = TimeDistributed(
            nn.Linear(num_features, hidden_size)
        )

        # Feature processing
        self.feature_grn = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout
        )

        # Temporal processing
        self.temporal_grn = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout
        )

        # Final processing
        self.final_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
        )

        # Final projection to single output
        self.output_projection = TimeDistributed(nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass
        Args:
            x: Input tensor of shape [batch_size, time_steps, features]
        Returns:
            Tensor of shape [batch_size, time_steps, 1]
        """
        # Project input
        hidden = self.input_projection(x)  # [batch, time, hidden]

        # Feature processing
        processed_features = self.feature_grn(hidden)  # [batch, time, hidden]

        # Temporal processing
        temporal_features = self.temporal_grn(
            processed_features
        )  # [batch, time, hidden]

        # Prepare for attention
        # Transpose to [time, batch, hidden] for attention
        query = temporal_features.transpose(0, 1)
        key = processed_features.transpose(0, 1)
        value = processed_features.transpose(0, 1)

        # Apply attention
        attended, _ = self.attention(query, key, value)
        # Back to [batch, time, hidden]
        attended = attended.transpose(0, 1)

        # Final processing
        processed = self.final_grn(attended)

        # Project to output
        output = self.output_projection(processed)  # [batch, time, 1]

        return output.squeeze(-1)  # [batch, time]


class TFTModel:
    def __init__(
        self,
        num_features: int,
        seq_length: int = 30,
        batch_size: int = 128,
        device: str = "cuda",
    ):
        self.model = TemporalFusionTransformer(
            num_features=num_features,
            hidden_size=256,  # Increased from 64
            num_heads=8,  # Increased from 4
        ).to(device)
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.device = device

    def create_sequences(
        self, X: torch.Tensor, y: torch.Tensor = None
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Create sequences

        Args:
            X: Input tensor with shape [time, features]
            y: Optional target tensor with shape [time]

        Returns:
            If y is None:
                X_seq: Tensor with shape [batch, time, features]
            If y is provided:
                tuple of (X_seq, y_seq) where:
                    X_seq: Tensor with shape [batch, time, features]
                    y_seq: Tensor with shape [batch]
        """
        X_seqs = []
        y_seqs = []

        for i in range(len(X) - self.seq_length + 1):
            X_seqs.append(X[i : i + self.seq_length])
            if y is not None:
                y_seqs.append(y[i + self.seq_length - 1])

        # Stack sequences
        X_seq = torch.stack(X_seqs, dim=0)

        if y is not None:
            # Stack y sequences
            y_seq = torch.stack(y_seqs, dim=0)
            return X_seq, y_seq
        return X_seq  # Return X_seq when y is None

    def train(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 400):
        """Train the model"""
        # Validate input dimensions
        if X.size(-1) != self.model.num_features:
            raise ValueError(
                f"Expected {self.model.num_features} features, got {X.size(-1)}"
            )

        # Create sequences
        X_seq, y_seq = self.create_sequences(X, y)

        # Create dataset (TensorDataset doesn't support named tensors)
        dataset = TensorDataset(X_seq, y_seq)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

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

        best_loss = float("inf")
        patience = 20
        patience_counter = 0
        training_start = time.time()
        last_log = training_start

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

                loss = peak_weighted_loss(predictions, batch_y)
                loss.backward()

                # Gradient clipping to prevent instability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                batch_count += 1

                # Free up memory
                del outputs, predictions, loss
                if batch_count % 10 == 0:  # Every 10 batches
                    torch.cuda.empty_cache()

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
        """Generate predictions"""
        # Validate input dimension
        if X.size(-1) != self.model.num_features:
            raise ValueError(
                f"Expected {self.model.num_features} features, got {X.size(-1)}"
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
        pad_predictions = np.full(self.seq_length - 1, predictions[0])
        final_predictions = np.concatenate([pad_predictions, predictions])

        return final_predictions
