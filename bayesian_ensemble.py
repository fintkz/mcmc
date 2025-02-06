import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.zeros(out_features, in_features))
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.zeros(out_features))
        
        # Better initialization
        self.weight_mu.data.normal_(0, 0.05)
        self.weight_rho.data.fill_(-3)
        self.bias_mu.data.normal_(0, 0.05)
        self.bias_rho.data.fill_(-3)
        
    def forward(self, x, sample=False):
        if sample:
            weight = self.weight_mu + torch.log1p(torch.exp(self.weight_rho)) * torch.randn_like(self.weight_mu)
            bias = self.bias_mu + torch.log1p(torch.exp(self.bias_rho)) * torch.randn_like(self.bias_mu)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(x, weight, bias)

class BayesianNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):  # Increased hidden dim
        super().__init__()
        self.hidden1 = BayesianLinear(input_dim, hidden_dim)
        self.hidden2 = BayesianLinear(hidden_dim, hidden_dim)  # Added second hidden layer
        self.output = BayesianLinear(hidden_dim, output_dim)
        
    def forward(self, x, sample=False):
        x = F.relu(self.hidden1(x, sample))
        x = F.relu(self.hidden2(x, sample))  # Added second hidden layer
        return self.output(x, sample)
    
    def loss(self, x, y, num_samples=5, kl_weight=0.1):  # Reduced KL weight
        # Ensure y has correct shape
        if len(y.shape) == 1:
            y = y.view(-1, 1)
        
        # Multiple forward passes
        outputs = torch.stack([self.forward(x, sample=True) for _ in range(num_samples)])
        mean_output = outputs.mean(0)
        
        # Negative log likelihood with proper shapes
        log_likelihood = -F.mse_loss(mean_output, y, reduction='sum')
        
        # KL divergence for all layers
        kl_div = 0
        for module in self.modules():
            if isinstance(module, BayesianLinear):
                kl_div += self._kl_divergence(module)
        
        return -log_likelihood + kl_weight * kl_div
    
    def _kl_divergence(self, layer):
        weight_std = torch.log1p(torch.exp(layer.weight_rho))
        bias_std = torch.log1p(torch.exp(layer.bias_rho))
        
        weight_posterior = dist.Normal(layer.weight_mu, weight_std)
        bias_posterior = dist.Normal(layer.bias_mu, bias_std)
        
        weight_prior = dist.Normal(torch.zeros_like(layer.weight_mu), torch.ones_like(layer.weight_mu))
        bias_prior = dist.Normal(torch.zeros_like(layer.bias_mu), torch.ones_like(layer.bias_mu))
        
        return (torch.sum(dist.kl_divergence(weight_posterior, weight_prior)) + 
                torch.sum(dist.kl_divergence(bias_posterior, bias_prior)))

class GPUBayesianEnsemble:
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, n_models=5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_models = n_models
        self.models = []
        
    def train(self, X, y, epochs=100, batch_size=32, num_samples=5):
        X = torch.FloatTensor(X).to(self.device)
        y = torch.FloatTensor(y).to(self.device)
        
        # Ensure y has correct shape
        if len(y.shape) == 1:
            y = y.view(-1, 1)
        
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for i in range(self.n_models):
            model = BayesianNeuralNetwork(
                self.input_dim, 
                self.hidden_dim, 
                self.output_dim
            ).to(self.device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
            
            best_loss = float('inf')
            for epoch in range(epochs):
                epoch_loss = 0
                for batch_X, batch_y in loader:
                    optimizer.zero_grad()
                    loss = model.loss(batch_X, batch_y, num_samples)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Added gradient clipping
                    optimizer.step()
                    epoch_loss += loss.item()
                
                scheduler.step(epoch_loss)
                
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                
                if (epoch + 1) % 20 == 0:
                    print(f"Model {i+1}/{self.n_models}, Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
            
            self.models.append(model)
    
    def predict(self, X, num_samples=100):
        X = torch.FloatTensor(X).to(self.device)
        predictions = []
        
        for model in self.models:
            model_preds = []
            with torch.no_grad():
                for _ in range(num_samples):
                    pred = model(X, sample=True)
                    model_preds.append(pred.cpu().numpy())
            predictions.append(np.mean(model_preds, axis=0))
        
        predictions = np.stack(predictions)
        mean_pred = np.mean(predictions, axis=0).flatten()
        std_pred = np.std(predictions, axis=0).flatten()
        
        return mean_pred, std_pred

def preprocess_data_gpu(df):
    # Create previous demand feature
    df['prev_demand'] = df['demand'].shift(1)
    df = df.dropna()
    
    # Select features
    feature_columns = ['prev_demand', 'temperature', 'is_weekend', 'is_holiday']
    X = df[feature_columns].values
    y = df['demand'].values
    
    # Scale features and target
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    return X_scaled, y_scaled, scaler_y
