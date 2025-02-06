import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np

def peak_weighted_loss(y_pred, y_true, alpha=2.0, beta=0.5):
    """Custom loss function that puts more emphasis on peaks"""
    if len(y_true.shape) == 1:
        y_true = y_true.view(-1, 1)
    if len(y_pred.shape) == 1:
        y_pred = y_pred.view(-1, 1)
        
    base_loss = F.mse_loss(y_pred, y_true, reduction='none')
    
    # Identify peaks and difficult points
    mean = y_true.mean()
    std = y_true.std()
    peak_mask = (y_true > (mean + beta * std)).float()
    
    # Additional weight for points where prediction is far from truth
    pred_error = torch.abs(y_pred - y_true)
    error_weight = 1.0 + torch.sigmoid(pred_error - pred_error.mean())
    
    # Combine weights
    total_weight = 1.0 + (alpha - 1.0) * peak_mask * error_weight
    
    return (base_loss * total_weight).mean()

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

class TFTModel:
    def __init__(self, num_features, hidden_size=512, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = TemporalFusionTransformer(
            num_features=num_features,
            hidden_size=hidden_size
        ).to(self.device)
        
    def train(self, X, y, epochs=400, batch_size=64):
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
            
        X = torch.FloatTensor(X).to(self.device)
        y = torch.FloatTensor(y).to(self.device)
        
        if len(y.shape) == 1:
            y = y.view(-1, 1, 1)
        elif len(y.shape) == 2:
            y = y.unsqueeze(-1)
            
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=0.01,
            epochs=epochs,
            steps_per_epoch=len(X) // batch_size + 1,
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
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
                loss = peak_weighted_loss(outputs, batch_y)
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
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
            
        X = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(X)
            
        return predictions.squeeze(-1).cpu().numpy()
