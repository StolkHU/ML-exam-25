import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock1D(nn.Module):
    """
    Een residual block voor 1D data met skip connections
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_batchnorm=True):
        super(ResidualBlock1D, self).__init__()
        
        self.use_batchnorm = use_batchnorm
        
        # Eerste conv laag
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=kernel_size//2)
        if use_batchnorm:
            self.bn1 = nn.BatchNorm1d(out_channels)
        
        # Tweede conv laag
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                              stride=1, padding=kernel_size//2)
        if use_batchnorm:
            self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection - als dimensies niet matchen
        self.skip_connection = None
        if stride != 1 or in_channels != out_channels:
            layers = [nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)]
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_channels))
            self.skip_connection = nn.Sequential(*layers)
    
    def forward(self, x):
        # Bewaar originele input voor skip connection
        identity = x
        
        # Eerste conv + batchnorm + relu
        out = self.conv1(x)
        if self.use_batchnorm:
            out = self.bn1(out)
        out = F.relu(out)
        
        # Tweede conv + batchnorm (geen relu hier!)
        out = self.conv2(out)
        if self.use_batchnorm:
            out = self.bn2(out)
        
        # Skip connection toepassen
        if self.skip_connection is not None:
            identity = self.skip_connection(x)
        
        # Element-wise optelling + relu
        out += identity
        out = F.relu(out)
        
        return out

class ConfigurableCNN1D(nn.Module):
    """
    Configureerbare 1D CNN waarbij je batchnorm en skip connections aan/uit kunt zetten
    """
    def __init__(self, config):
        super(ConfigurableCNN1D, self).__init__()
        
        # Config parameters
        self.hidden = config.get('hidden', 64)
        self.dropout = config.get('dropout', 0.1)
        self.output = config.get('output', 5)
        self.use_batchnorm = config.get('use_batchnorm', True)
        self.use_skip_connections = config.get('use_skip_connections', True)
        self.num_blocks = config.get('num_blocks', 3)
        
        # Input conv laag
        self.input_conv = nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3)
        if self.use_batchnorm:
            self.input_bn = nn.BatchNorm1d(32)
        
        # Residual blocks of gewone conv blocks
        self.blocks = nn.ModuleList()
        
        if self.use_skip_connections:
            # Met skip connections (ResNet-style)
            channels = [32, 64, 128, self.hidden]
            for i in range(self.num_blocks):
                in_ch = channels[i]
                out_ch = channels[i + 1] if i + 1 < len(channels) else self.hidden
                stride = 2 if i > 0 else 1  # Downsample na eerste block
                
                block = ResidualBlock1D(in_ch, out_ch, kernel_size=3, 
                                      stride=stride, use_batchnorm=self.use_batchnorm)
                self.blocks.append(block)
        else:
            # Zonder skip connections (gewone conv blocks)
            channels = [32, 64, 128, self.hidden]
            for i in range(self.num_blocks):
                in_ch = channels[i]
                out_ch = channels[i + 1] if i + 1 < len(channels) else self.hidden
                stride = 2 if i > 0 else 1
                
                layers = []
                layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=3, 
                                      stride=stride, padding=1))
                if self.use_batchnorm:
                    layers.append(nn.BatchNorm1d(out_ch))
                layers.append(nn.ReLU())
                
                block = nn.Sequential(*layers)
                self.blocks.append(block)
        
        # Global pooling en classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden, self.hidden // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden // 2, self.output)
        )
    
    def forward(self, x):
        # Input shape aanpassen
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = x.transpose(1, 2)  # (batch, channels, seq_len)
        
        # Input conv
        x = self.input_conv(x)
        if self.use_batchnorm:
            x = self.input_bn(x)
        x = F.relu(x)
        
        # Door alle blocks
        for block in self.blocks:
            x = block(x)
        
        # Global pooling
        x = self.global_pool(x)  # (batch, hidden, 1)
        x = x.squeeze(-1)  # (batch, hidden)
        
        # Classifier
        x = self.dropout_layer(x)
        x = self.classifier(x)
        
        return x

class SimpleConfigurableCNN1D(nn.Module):
    """
    Eenvoudigere versie - makkelijker te begrijpen
    """
    def __init__(self, config):
        super(SimpleConfigurableCNN1D, self).__init__()
        
        self.output = config.get('output', 5)
        self.use_batchnorm = config.get('use_batchnorm', True)
        self.dropout_rate = config.get('dropout', 0.1)
        
        # Conv lagen
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        
        # Optionele batch normalization
        if self.use_batchnorm:
            self.bn1 = nn.BatchNorm1d(32)
            self.bn2 = nn.BatchNorm1d(64)
            self.bn3 = nn.BatchNorm1d(128)
        
        # Pooling en classifier
        self.pool = nn.MaxPool1d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(4)
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, self.output)
        )
    
    def forward(self, x):
        # Input shape aanpassen
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = x.transpose(1, 2)
        
        # Conv blok 1
        x = self.conv1(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Conv blok 2
        x = self.conv2(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Conv blok 3
        x = self.conv3(x)
        if self.use_batchnorm:
            x = self.bn3(x)
        x = F.relu(x)
        
        # Pooling en flatten
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.classifier(x)
        return x

