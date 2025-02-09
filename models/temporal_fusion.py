import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from utils import peak_weighted_loss

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super().__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        
        x_reshaped = x.contiguous().view(-1, x.size(-1))
        y = self.module(x_reshaped)
        
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            y = y.view(-1, x.size(1), y.size(-1))
        return y

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        self.fc1 = TimeDistributed(nn.Linear(input_size, hidden_size))
        self.fc2 = TimeDistributed(nn.Linear(hidden_size, hidden_size))
        self.fc3 = TimeDistributed(nn.Linear(hidden_size, output_size))
        
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(output_size)
        
        self.gate = TimeDistributed(nn.Linear(input_size + output_size, output_size))
        
        if input_size != output_size:
            self.skip_layer = TimeDistributed(nn.Linear(input_size, output_size))
        else:
            self.skip_layer = None
            
    def forward(self, x):
        # Main branch
        h = F.elu(self.fc1(x))
        h = self.dropout(h)
        h = F.elu(self.fc2(h))
        h = self.dropout(h)
        h = self.fc3(h)
        h = self.dropout(h)
        
        # Skip connection
        if self.skip_layer is not None:
            x = self.skip_layer(x)
            
        # Gating mechanism
        gate = self.gate(torch.cat([x, h], dim=-1))
        gate = torch.sigmoid(gate)
        
        output = x + gate * h
        return self.layernorm(output)

class TemporalFusionTransformer(nn.Module):
    def __init__(
        self,
        num_features,
        hidden_size=512,
        num_heads=8,
        num_encoder_layers=4,
        dropout=0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features
        
        # Initial processing
        self.input_layer = TimeDistributed(nn.Linear(num_features, hidden_size))
        
        # Feature processing
        self.feature_grn = GatedResidualNetwork(
            hidden_size,
            hidden_size * 2,
            hidden_size,
            dropout
        )
        
        # Position encoding
        self.position_encoding = nn.Parameter(torch.randn(1, 1000, hidden_size))  # Max sequence length of 1000
        
        # Multi-head self-attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_encoder_layers)
        ])
        
        # Layer normalization and skip connections
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size)
            for _ in range(num_encoder_layers)
        ])
        
        # Feed-forward layers after attention
        self.ff_layers = nn.ModuleList([
            GatedResidualNetwork(
                hidden_size,
                hidden_size * 2,
                hidden_size,
                dropout
            )
            for _ in range(num_encoder_layers)
        ])
        
        # Output processing
        self.pre_output_grn = GatedResidualNetwork(
            hidden_size,
            hidden_size * 2,
            hidden_size,
            dropout
        )
        self.output_layer = TimeDistributed(nn.Linear(hidden_size, 1))
        
    def forward(self, x):
        # Initial processing
        x = self.input_layer(x)
        
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.position_encoding[:, :seq_len, :]
        
        # Feature processing
        x = self.feature_grn(x)
        
        # Self-attention blocks with skip connections and feed-forward
        for attention, norm, ff_layer in zip(self.attention_layers, self.layer_norms, self.ff_layers):
            # Multi-head attention
            attended, _ = attention(x, x, x)
            x = norm(x + attended)
            
            # Feed-forward with residual
            x = x + ff_layer(x)
        
        # Output processing
        x = self.pre_output_grn(x)
        return self.output_layer(x)

def create_sequences(X, y, seq_length=30):
    """Create sequences for temporal modeling"""
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length])
    return np.array(Xs), np.array(ys)

class TFTModel:
    def __init__(self, num_features, hidden_size=512, seq_length=30, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.seq_length = seq_length
        self.model = TemporalFusionTransformer(
            num_features=num_features,
            hidden_size=hidden_size
        ).to(self.device)
        
    def train(self, X, y, epochs=400, batch_size=64):
        # Create sequences for temporal modeling
        X_seq, y_seq = create_sequences(X, y, self.seq_length)
        
        X = torch.FloatTensor(X_seq).to(self.device)
        y = torch.FloatTensor(y_seq).to(self.device)
        
        if len(y.shape) == 1:
            y = y.unsqueeze(-1)
            
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=0.001, 
            weight_decay=0.01
        )
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=0.01,
            epochs=epochs,
            steps_per_epoch=max(1, len(X) // batch_size),
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        best_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            self.model.train()
            
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                # Take the last prediction for each sequence
                predictions = outputs[:, -1, :]
                loss = peak_weighted_loss(predictions, batch_y)
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
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    
    def predict(self, X):
        # Create sequences for prediction
        X_seq = []
        for i in range(len(X) - self.seq_length + 1):
            X_seq.append(X[i:(i + self.seq_length)])
        X_seq = np.array(X_seq)
        
        X = torch.FloatTensor(X_seq).to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(X)
            # Take the last prediction for each sequence
            predictions = predictions[:, -1, 0]
            
        # For the first seq_length-1 points, use the first prediction
        pad_predictions = np.full(self.seq_length - 1, predictions[0])
        final_predictions = np.concatenate([pad_predictions, predictions.cpu().numpy()])
            
        return final_predictions