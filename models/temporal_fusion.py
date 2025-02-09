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
        
        # Main layers
        self.fc1 = TimeDistributed(nn.Linear(input_size, hidden_size))
        self.elu1 = nn.ELU()
        self.fc2 = TimeDistributed(nn.Linear(hidden_size, output_size))
        self.elu2 = nn.ELU()
        
        # Skip connection if input and output dimensions differ
        self.skip = TimeDistributed(nn.Linear(input_size, output_size)) if input_size != output_size else None
        
        # Gating mechanism
        self.gate = TimeDistributed(nn.Linear(input_size + output_size, output_size))
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_size)
        
    def forward(self, x):
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
        gate = torch.sigmoid(self.gate(torch.cat([x, h], dim=-1)))
        output = gate * h + (1 - gate) * x
        
        # Layer normalization
        output = self.layer_norm(output)
        
        return output

class TemporalFusionTransformer(nn.Module):
    def __init__(self, num_features, hidden_size=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        
        # Feature processing
        self.feature_grn = GatedResidualNetwork(
            input_size=num_features,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout
        )
        
        # Temporal processing
        self.temporal_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout
        )
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Final processing
        self.final_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=1,  # Single output value
            dropout=dropout
        )
    
    def forward(self, x):
        # Feature processing
        processed_features = self.feature_grn(x)
        
        # Temporal processing
        temporal_features = self.temporal_grn(processed_features)
        
        # Self-attention
        attended_features, _ = self.attention(
            temporal_features,
            temporal_features,
            temporal_features
        )
        
        # Final processing
        output = self.final_grn(attended_features)
        
        return output
    
def create_sequences(X, y, seq_length):
    """Create sequences for temporal modeling"""
    Xs, ys = [], []
    for i in range(len(X) - seq_length + 1):
        Xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length - 1])
    return np.array(Xs), np.array(ys)

class TFTModel:
    def __init__(
        self,
        num_features,
        seq_length=30,
        hidden_size=64,  # Reduced from 512
        num_heads=4,     # Reduced from 8
        dropout=0.1,
        batch_size=512,
        device=None
    ):
        self.num_features = num_features
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = TemporalFusionTransformer(
            num_features=num_features,
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        ).to(self.device)
    
    def train(self, X, y, epochs=400):
        """Train the model"""
        # Create sequences for temporal modeling
        X_seq = []
        y_seq = []
        
        for i in range(len(X) - self.seq_length + 1):
            X_seq.append(X[i:(i + self.seq_length)])
            y_seq.append(y[i + self.seq_length - 1])
            
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # Keep data on CPU initially
        X = torch.FloatTensor(X_seq)
        y = torch.FloatTensor(y_seq)
        
        if len(y.shape) == 1:
            y = y.unsqueeze(-1)
        
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=False
        )
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.01,
            epochs=epochs,
            steps_per_epoch=len(loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        best_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            self.model.train()
            
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                predictions = outputs[:, -1, 0]  # Take last timestep prediction
                
                loss = peak_weighted_loss(predictions, batch_y.squeeze())
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
        """Generate predictions"""
        # Create sequences for prediction
        X_seq = []
        for i in range(len(X) - self.seq_length + 1):
            X_seq.append(X[i:(i + self.seq_length)])
        X_seq = np.array(X_seq)
        
        # Keep data on CPU initially
        X = torch.FloatTensor(X_seq)
        self.model.eval()
        
        with torch.no_grad():
            # Process in batches
            predictions = []
            for i in range(0, len(X), self.batch_size):
                batch_X = X[i:i + self.batch_size].to(self.device)
                batch_preds = self.model(batch_X)[:, -1, 0]  # Take last timestep
                predictions.extend(batch_preds.cpu().numpy())
            
        # For the first seq_length-1 points, use the first prediction
        pad_predictions = np.full(self.seq_length - 1, predictions[0])
        final_predictions = np.concatenate([pad_predictions, predictions])
        
        return final_predictions