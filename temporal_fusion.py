import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super().__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
            
        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  
        y = self.module(x_reshape)
        
        # Reshape back
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  
        else:
            y = y.view(-1, x.size(1), y.size(-1))  
        return y

class GatedLinearUnit(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        
        self.fc = TimeDistributed(nn.Linear(input_size, hidden_size * 2))
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc(x)
        x = self.dropout_layer(x)
        return F.glu(x, dim=-1)

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        
        self.fc1 = TimeDistributed(nn.Linear(input_size, hidden_size))
        self.elu = nn.ELU()
        self.fc2 = TimeDistributed(nn.Linear(hidden_size, output_size))
        self.dropout_layer = nn.Dropout(dropout)
        
        self.gate = TimeDistributed(nn.Linear(input_size + output_size, output_size))
        
        if input_size != output_size:
            self.skip_layer = TimeDistributed(nn.Linear(input_size, output_size))
        else:
            self.skip_layer = None
            
    def forward(self, x, c=None):
        if c is not None:
            x = torch.cat([x, c], dim=-1)
            
        # Main branch
        h = self.fc1(x)
        h = self.elu(h)
        h = self.fc2(h)
        h = self.dropout_layer(h)
        
        # Skip connection
        if self.skip_layer is not None:
            x = self.skip_layer(x)
            
        # Gating mechanism
        gate = self.gate(torch.cat([x, h], dim=-1))
        gate = torch.sigmoid(gate)
        
        return x + gate * h

class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size, num_inputs, hidden_size, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_inputs = num_inputs
        
        self.flattened_grn = GatedResidualNetwork(
            input_size * num_inputs,
            hidden_size,
            num_inputs,
            dropout
        )
        
        self.single_variable_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_size,
                hidden_size,
                hidden_size,
                dropout
            ) for _ in range(num_inputs)
        ])
        
    def forward(self, x):
        # x should be of shape (batch_size, time_steps, num_inputs, input_size)
        batch_size = x.size(0)
        time_steps = x.size(1)
        
        # Flatten and apply GRN to get variable weights
        flat_x = x.view(batch_size, time_steps, -1)
        sparse_weights = self.flattened_grn(flat_x)
        sparse_weights = torch.softmax(sparse_weights, dim=-1).unsqueeze(-1)
        
        # Apply GRN to each variable
        processed_x = []
        for i in range(self.num_inputs):
            processed_x.append(self.single_variable_grns[i](x[..., i, :]))
            
        processed_x = torch.stack(processed_x, dim=-2)
        
        # Combine with weights
        return torch.sum(processed_x * sparse_weights, dim=-2)

class TemporalFusionTransformer(nn.Module):
    def __init__(
        self,
        num_features,
        hidden_size=128,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features
        
        # Variable selection networks
        self.feature_selection = VariableSelectionNetwork(
            input_size=1,  # Single value per feature
            num_inputs=num_features,
            hidden_size=hidden_size,
            dropout=dropout
        )
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Final output layers
        self.pre_output_grn = GatedResidualNetwork(
            hidden_size,
            hidden_size,
            hidden_size,
            dropout
        )
        self.output_layer = TimeDistributed(nn.Linear(hidden_size, 1))
        
    def forward(self, x):
        # x shape: (batch_size, time_steps, num_features)
        
        # Reshape input for variable selection
        x = x.unsqueeze(-1)  # Add feature dimension
        
        # Apply variable selection
        x = self.feature_selection(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Final processing
        x = self.pre_output_grn(x)
        outputs = self.output_layer(x)
        
        return outputs.squeeze(-1)

class TFTModel:
    def __init__(self, num_features, hidden_size=128, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = TemporalFusionTransformer(num_features, hidden_size=hidden_size).to(self.device)
        
    def train(self, X, y, epochs=100, batch_size=32):
        X = torch.FloatTensor(X).to(self.device)
        y = torch.FloatTensor(y).to(self.device)
        
        if len(y.shape) == 1:
            y = y.view(-1, 1)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=5, factor=0.5
        )
        
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        best_loss = float('inf')
        for epoch in range(epochs):
            epoch_loss = 0
            self.model.train()
            
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = F.mse_loss(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            
            scheduler.step(epoch_loss)
            
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    
    def predict(self, X):
        X = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(X)
            
        return predictions.cpu().numpy()
