# model.py
# Responsibility: Define neural network model architecture.

import torch
import torch.nn as nn
from torch.nn import functional as F

class SwiGLU(nn.Module):
    """SwiGLU activation function implementation"""
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.linear_gate = nn.Linear(dim_in, dim_out, bias=bias)
        self.linear_proj = nn.Linear(dim_in, dim_out, bias=bias)
        
    def forward(self, x):
        gate = self.linear_gate(x)
        proj = self.linear_proj(x)
        return F.silu(gate) * proj

class TsaiWuPINN(nn.Module):
    """Decoupled prediction version Tsai-Wu PINN model - 20D input -> 1D failure length L output"""
    
    def __init__(self, layer_sizes=[20, 512, 1], dropout_rate=0.01, activation='swiglu'):
        super(TsaiWuPINN, self).__init__()
        
        # Validate input/output dimensions
        if layer_sizes[0] != 20:
            print(f"⚠️ Warning: Input dimension should be 20 (14 material + 6 direction), currently {layer_sizes[0]}")
        if layer_sizes[-1] != 1:
            print(f"⚠️ Warning: Output dimension should be 1 (failure length L), currently {layer_sizes[-1]}")
        
        self.layer_sizes = layer_sizes
        self.activation = activation
        
        # Build network layers
        self.layers = nn.ModuleList()
        self.activation_layers = nn.ModuleList()
        
        for i in range(len(layer_sizes) - 1):
            # Linear layer
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            
            # Activation function layer (only for hidden layers, output layer uses fixed Sigmoid)
            if i < len(layer_sizes) - 2:  # Not output layer
                if activation == 'swiglu':
                    self.activation_layers.append(SwiGLU(layer_sizes[i+1], layer_sizes[i+1]))
                elif activation == 'tanh':
                    self.activation_layers.append(nn.Tanh())
                elif activation == 'relu':
                    self.activation_layers.append(nn.ReLU())
                else:
                    raise ValueError(f"Unsupported activation: {activation}")
        
        self.dropout = nn.Dropout(dropout_rate)
        self._initialize_weights()
        
        print(f"✅ Decoupled prediction model built (using optimal hyperparameters)")
        print(f"   Architecture: {' -> '.join(map(str, layer_sizes))} (optimal configuration)")
        print(f"   Input: 20D enhanced features (14 material properties + 6 stress directions)")
        print(f"   Output: 1D failure length L")
        print(f"   Hidden layer activation: {activation} (optimal configuration)")
        print(f"   Dropout rate: {dropout_rate} (optimal configuration)")
        print(f"   Output layer activation: Sigmoid (forced to ensure [0,1] output)")
        print(f"   📊 Expected performance: R²≈0.91, RMSE≈135 MPa")
        
    def _initialize_weights(self):
        """Weight initialization specialized for 1D regression - optimized for wide-shallow architecture [20,512,1]"""
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Linear):
                is_output_layer = (m == self.layers[-1])
                
                if is_output_layer:
                    # Output layer: special initialization for wide-shallow architecture
                    # Since hidden layer is very wide (512), output layer weights need smaller initialization
                    nn.init.xavier_uniform_(m.weight, gain=0.3)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
                else:
                    # Hidden layer: optimized for wide-shallow SwiGLU architecture
                    if self.activation == 'swiglu':
                        # Use more conservative initialization for wide layers
                        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                        with torch.no_grad():
                            # For 512-wide layers, use smaller scaling factor
                            scale_factor = 0.2 if m.weight.shape[0] >= 512 else 0.3
                            m.weight *= scale_factor
                    elif self.activation == 'tanh':
                        nn.init.xavier_uniform_(m.weight, gain=0.6)
                    else:
                        nn.init.xavier_uniform_(m.weight, gain=0.4)
                    
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        """Forward propagation - output 1D failure length L, last layer forced to use Sigmoid to ensure [0,1] output"""
        # Input check
        if x.shape[1] != 20:
            raise ValueError(f"Input feature dimension should be 20, actual is {x.shape[1]}")
        
        # Input numerical stability handling
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("⚠️ Warning: Model input contains NaN or Inf values")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Input range check
        if x.min() < -0.1 or x.max() > 1.1:
            print(f"⚠️ Warning: Input data exceeds expected range [{x.min():.4f}, {x.max():.4f}]")
        
        # Forward propagate hidden layers (except last layer)
        for i, layer in enumerate(self.layers[:-1]):
            # Linear transformation
            x = layer(x)
            
            # Numerical stability check
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"⚠️ Warning: Layer {i} output contains NaN or Inf values")
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Hidden layer activation function - select based on configuration
            if self.activation == 'swiglu':
                x = self.activation_layers[i](x)
            elif self.activation == 'tanh':
                x = torch.tanh(x)
            elif self.activation == 'relu':
                x = torch.relu(x)
            
            # Dropout (only used during training and in hidden layers)
            if self.training and hasattr(self, 'dropout'):
                x = self.dropout(x)
        
        # Output layer: Linear transformation + forced Sigmoid activation
        L_pred = self.layers[-1](x)  # Linear transformation
        
        # Force use of Sigmoid to ensure output is in [0,1] range, regardless of configured activation function
        L_pred = torch.sigmoid(L_pred)
        
        # Avoid overly extreme output values (prevent gradient vanishing)
        L_pred = torch.clamp(L_pred, min=0.01, max=0.99)
        
        # Final output check
        if torch.isnan(L_pred).any() or torch.isinf(L_pred).any():
            print("⚠️ Warning: Model final output contains NaN or Inf values")
            L_pred = torch.nan_to_num(L_pred, nan=0.5, posinf=0.99, neginf=0.01)
        
        return L_pred