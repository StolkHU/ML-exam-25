# model.py - Getuned ModularCNN op basis van beste 96.7% config
import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExciteBlock(nn.Module):
    """Improved Squeeze-and-Excitation block for 1D CNNs"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        
    def forward(self, x):
        b, c, _ = x.size()
        y = self.pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1)
        return x * y

class ResidualBlock(nn.Module):
    """Residual connection for deeper networks - geoptimaliseerd"""
    def __init__(self, in_channels, out_channels, kernel_size, use_se=False):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection - alleen als channels veranderen
        self.skip_connection = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        # Alleen SE blocks (attention blocks weggehaald)
        self.se = SqueezeExciteBlock(out_channels) if use_se else None
        
    def forward(self, x):
        identity = self.skip_connection(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.se:
            out = self.se(out)
            
        out += identity
        return F.relu(out)

class ModularCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_channels = config["input_channels"]
        self.output = config["output"]
        self.dropout = config.get("dropout", 0.42)  # Default naar beste waarde
        self.squeeze_excite = config.get("squeeze_excite", False)
        
        
        self.conv_layers_config = config["conv_layers"]
        self.fc1_size = config["fc1_size"]
        self.fc2_size = config["fc2_size"]
        self.fc3_size = config["fc3_size"]
        
        # Use residual connections for deeper networks (behouden)
        self.use_residual = config.get("use_residual", len(self.conv_layers_config) > 3)
        
        # Build improved architecture
        self.conv_layers = self._build_conv_layers()
        
        # Get flatten dimension with dummy forward pass
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, 500)
            dummy_output = self._forward_conv_layers(dummy_input)
            self.flatten_dim = dummy_output.view(1, -1).size(1)
        
        # Improved FC layers with batch normalization
        self.fc1 = nn.Linear(self.flatten_dim, self.fc1_size)
        self.bn_fc1 = nn.BatchNorm1d(self.fc1_size)
        self.dropout1 = nn.Dropout(self.dropout)
        
        self.fc2 = nn.Linear(self.fc1_size, self.fc2_size)
        self.bn_fc2 = nn.BatchNorm1d(self.fc2_size)
        self.dropout2 = nn.Dropout(self.dropout)
        
        self.fc3 = nn.Linear(self.fc2_size, self.fc3_size)
        self.bn_fc3 = nn.BatchNorm1d(self.fc3_size)
        self.dropout3 = nn.Dropout(self.dropout * 0.5)  # Less dropout before final layer
        
        self.output_layer = nn.Linear(self.fc3_size, self.output)
        
        # Initialize weights
        self._initialize_weights()
        
    def _build_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.input_channels
        
        # Skip layers mechanisme VOLLEDIG weggehaald
        for i, layer_cfg in enumerate(self.conv_layers_config):
            out_channels = layer_cfg["out_channels"]
            kernel_size = layer_cfg["kernel_size"]
            
            if self.use_residual and i > 0:  # Use residual blocks for deeper layers
                block = ResidualBlock(
                    in_channels, out_channels, kernel_size,
                    use_se=self.squeeze_excite
                )
            else:
                # Standard conv block for first layer
                block = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True)
                )
            
            layers.append(block)
            
            # Add pooling
            pool_type = layer_cfg.get("pool", "none")
            if pool_type == "max":
                layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            elif pool_type == "avg":
                layers.append(nn.AvgPool1d(kernel_size=2, stride=2))
            
            in_channels = out_channels
            
        return layers
    
    def _forward_conv_layers(self, x):
        """Forward pass through conv layers only"""
        for layer in self.conv_layers:
            x = layer(x)
        return x
    
    def _initialize_weights(self):
        """Improved weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Handle input dimensions
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.size(1) != self.input_channels:
            x = x.transpose(1, 2)
        
        # Conv layers
        x = self._forward_conv_layers(x)
        
        # Flatten and FC layers with improved normalization
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn_fc3(self.fc3(x)))
        x = self.dropout3(x)
        
        return self.output_layer(x)
    
    def predict_proba(self, x):
        """Probability predictions"""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)