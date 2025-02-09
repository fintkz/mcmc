import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import TensorDataset, DataLoader
from utils import peak_weighted_loss
import numpy as np
from typing import Union


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply module to named tensor preserving batch and time dimensions"""
        # Input shape: [batch, time, features] or [batch, features]
        x = x.refine_names('batch', ..., 'features')
        
        if x.names.count(None) == 0:  # All dimensions are named
            # Log shapes for debugging
            print(f"TimeDistributed input shape: {x.shape}")
            print(f"Module weight shape: {self.module.weight.shape}")
            
            # Flatten batch and time for module application
            x_reshape = x.align_to('batch', 'time', 'features').rename(None)
            x_reshape = x_reshape.reshape(-1, x_reshape.size(-1))
            print(f"Reshaped input shape: {x_reshape.shape}")
            
            y = self.module(x_reshape)
            
            # Restore batch and time dimensions
            batch_size = x.size('batch')
            time_size = x.size('time')
            y = y.reshape(batch_size, time_size, -1)
            return y.refine_names('batch', 'time', 'features')
        else:
            return self.module(x.rename(None)).refine_names('batch', 'features')


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.1):
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
        self.skip = TimeDistributed(nn.Linear(input_size, output_size)) if input_size != output_size else None

        # Gating mechanism - FIXED: input size should be input_size + output_size
        self.gate = TimeDistributed(nn.Linear(input_size + output_size, output_size))
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with named tensors"""
        # Input shape: [batch, time, features]
        x = x.refine_names('batch', 'time', 'features')
        
        # Validate input dimension
        if x.size('features') != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, got {x.size('features')}")

        # Main branch
        h = self.fc1(x)
        h = self.elu1(h.rename(None)).refine_names('batch', 'time', 'features')
        h = self.dropout(h.rename(None)).refine_names('batch', 'time', 'features')
        h = self.fc2(h)
        h = self.elu2(h.rename(None)).refine_names('batch', 'time', 'features')

        # Skip connection
        if self.skip is not None:
            x = self.skip(x)

        # Gating mechanism
        combined = torch.cat([x, h], dim='features')
        gate = torch.sigmoid(self.gate(combined).rename(None)).refine_names('batch', 'time', 'features')
        output = gate * h + (1 - gate) * x

        # Layer normalization
        output = self.layer_norm(output.rename(None)).refine_names('batch', 'time', 'features')
        return output


class TemporalFusionTransformer(nn.Module):
    def __init__(self, num_features: int, hidden_size: int = 64, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size

        if hidden_size % num_heads != 0:
            raise ValueError(f"Hidden size ({hidden_size}) must be divisible by num_heads ({num_heads})")

        # Project input to hidden size
        self.input_projection = TimeDistributed(nn.Linear(num_features, hidden_size))

        # Feature processing
        self.feature_grn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)

        # Temporal processing
        self.temporal_grn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)

        # Final processing
        self.final_grn = GatedResidualNetwork(hidden_size, hidden_size, 1, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with named tensors"""
        # Input shape: [batch, time, features]
        x = x.refine_names('batch', 'time', 'features')
        
        # Validate input dimension
        if x.size('features') != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, got {x.size('features')}")

        # Project input
        projected = self.input_projection(x)

        # Feature processing
        processed = self.feature_grn(projected)

        # Temporal processing
        temporal = self.temporal_grn(processed)

        # Self-attention (requires unnamed tensors)
        attended, _ = self.attention(
            temporal.rename(None).permute(1, 0, 2),  # [time, batch, hidden]
            temporal.rename(None).permute(1, 0, 2),
            temporal.rename(None).permute(1, 0, 2)
        )
        attended = attended.permute(1, 0, 2).refine_names('batch', 'time', 'features')

        # Final processing
        output = self.final_grn(attended)
        return output


class TFTModel:
    def __init__(self, num_features: int, seq_length: int = 30, batch_size: int = 32, device: str = 'cuda'):
        self.model = TemporalFusionTransformer(
            num_features=num_features,
            hidden_size=64,
            num_heads=4
        ).to(device)
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.device = device

    def create_sequences(self, X: torch.Tensor, y: torch.Tensor = None) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Create sequences with named dimensions
        
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
        # Input shapes: X[time, features], y[time]
        X = X.refine_names('time', 'features')
        if y is not None:
            y = y.refine_names('time')

        X_seqs = []
        y_seqs = []

        for i in range(len(X) - self.seq_length + 1):
            # Drop names before appending
            X_seqs.append(X[i:i + self.seq_length].rename(None))
            if y is not None:
                y_seqs.append(y[i + self.seq_length - 1].rename(None))

        # Stack sequences without names, then restore names
        X_seq = torch.stack(X_seqs, dim=0).refine_names('batch', 'time', 'features')

        if y is not None:
            # Stack y sequences without names, then restore names
            y_seq = torch.stack(y_seqs, dim=0).refine_names('batch')
            return X_seq, y_seq
        return X_seq  # Return X_seq when y is None

    def train(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 400):
        """Train the model with named tensors"""
        # Ensure inputs have proper names
        X = X.refine_names('time', 'features')
        y = y.refine_names('time')
        
        # Validate input dimensions
        if X.size('features') != self.model.num_features:
            raise ValueError(
                f"Expected {self.model.num_features} features, got {X.size('features')}"
            )

        # Create sequences
        X_seq, y_seq = self.create_sequences(X, y)
        
        # Create dataset (TensorDataset doesn't support named tensors)
        dataset = TensorDataset(X_seq.rename(None), y_seq.rename(None))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=0.01,
            epochs=epochs,
            steps_per_epoch=len(loader),
            pct_start=0.3,
            anneal_strategy="cos"
        )

        best_loss = float('inf')
        patience = 20
        patience_counter = 0

        for epoch in range(epochs):
            epoch_loss = 0
            self.model.train()

            for batch_X, batch_y in loader:
                # Restore names for tensors
                batch_X = batch_X.to(self.device).refine_names('batch', 'time', 'features')
                batch_y = batch_y.to(self.device).refine_names('batch')

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                predictions = outputs[:, -1, 0]  # Take last timestep prediction

                loss = peak_weighted_loss(predictions.rename(None), batch_y.rename(None))
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
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
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Generate predictions with named tensors"""
        # Ensure input has proper names
        X = X.refine_names('time', 'features')
        
        # Validate input dimension
        if X.size('features') != self.model.num_features:
            raise ValueError(
                f"Expected {self.model.num_features} features, got {X.size('features')}"
            )

        # Create sequences
        X_seq = self.create_sequences(X)
        self.model.eval()

        with torch.no_grad():
            predictions = []
            # Process in batches
            for i in range(0, len(X_seq), self.batch_size):
                batch_X = X_seq[i:i + self.batch_size].to(self.device)
                batch_preds = self.model(batch_X)[:, -1, 0]  # Take last timestep
                predictions.extend(batch_preds.cpu().numpy())

        # For the first seq_length-1 points, use the first prediction
        pad_predictions = np.full(self.seq_length - 1, predictions[0])
        final_predictions = np.concatenate([pad_predictions, predictions])

        return final_predictions