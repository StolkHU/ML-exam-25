from dataclasses import asdict, dataclass

import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(
        self, features: int, num_classes: int, units1: int, units2: int
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.units1 = units1
        self.units2 = units2
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(features, units1),
            nn.ReLU(),
            nn.Linear(units1, units2),
            nn.ReLU(),
            nn.Linear(units2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class CNN(nn.Module):
    def __init__(
        self,
        features: int,
        num_classes: int,
        kernel_size: int,
        filter1: int,
        filter2: int,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.filter1 = filter1
        self.filter2 = filter2

        self.convolutions = nn.Sequential(
            nn.Conv2d(features, filter1, kernel_size=kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(filter1, filter2, kernel_size=kernel_size, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(filter2, 32, kernel_size=kernel_size, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolutions(x)
        logits = self.dense(x)
        return logits


@dataclass
class CNNConfig:
    matrixshape: tuple
    batchsize: int
    input_channels: int
    hidden: int
    kernel_size: int
    maxpool: int
    num_layers: int
    num_classes: int
    dropout: float = 0.0
    batch_norm: bool = False
    use_skip_connections: bool = False


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, batch_norm=False):  # ADRIAAN Added batch_norm parameter
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.batch_norm = batch_norm  # ADRIAAN Store batch_norm flag
        
        if batch_norm:  # ADRIAAN Add conditional batch normalization
            self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:  # ADRIAAN Apply batch norm if enabled
            x = self.bn(x)
        return x


class ResidualBlock(nn.Module):  # ADRIAAN Added entire ResidualBlock class for skip connections
    """Residual block met skip connection"""
    def __init__(self, channels, kernel_size, batch_norm=False, dropout=0.0):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, kernel_size, batch_norm)
        self.relu1 = nn.ReLU()
        self.conv2 = ConvBlock(channels, channels, kernel_size, batch_norm)
        self.relu2 = nn.ReLU()
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None  # ADRIAAN Added dropout to residual blocks
        
    def forward(self, x):
        # Skip connection: bewaar originele input
        identity = x  # ADRIAAN Store input for skip connection
        
        # Eerste conv + relu
        out = self.conv1(x)
        out = self.relu1(out)
        
        # Dropout na eerste conv
        if self.dropout:  # ADRIAAN Apply dropout if configured
            out = self.dropout(out)
        
        # Tweede conv (geen relu hier)
        out = self.conv2(out)
        
        # Skip connection: voeg originele input toe
        out = out + identity  # ADRIAAN Add skip connection
        
        # ReLU na de skip connection
        out = self.relu2(out)
        
        return out


class CNNblocks(nn.Module):
    def __init__(self, config: CNNConfig) -> None:
        super().__init__()
        self.config = asdict(config)
        self.use_skip_connections = config.use_skip_connections  # ADRIAAN Store skip connections flag
        
        # Eerste conv layer (kan niet skip hebben omdat input channels anders zijn)
        self.first_conv = ConvBlock(config.input_channels, config.hidden, config.kernel_size, 
                                   batch_norm=config.batch_norm)  # ADRIAAN Added batch_norm to first conv
        self.first_relu = nn.ReLU()
        
        # Bouw de layers
        self.layers = nn.ModuleList()  # ADRIAAN Changed from convolutions to layers for flexibility
        pool = config.maxpool
        num_maxpools = 0
        
        for i in range(config.num_layers):
            if self.use_skip_connections:  # ADRIAAN Added conditional skip connections
                # Gebruik residual blocks
                self.layers.append(
                    ResidualBlock(config.hidden, config.kernel_size, 
                                config.batch_norm, config.dropout)  # ADRIAAN Use ResidualBlock for skip connections
                )
            else:
                # Normale conv blocks
                layer_seq = [
                    ConvBlock(config.hidden, config.hidden, config.kernel_size,
                             batch_norm=config.batch_norm),  # ADRIAAN Added batch_norm parameter
                    nn.ReLU()
                ]
                
                if config.dropout > 0:  # ADRIAAN Added conditional dropout
                    layer_seq.append(nn.Dropout2d(config.dropout))
                
                self.layers.append(nn.Sequential(*layer_seq))  # ADRIAAN Create sequential layer
            
            # Maxpool elke 2e layer
            if i % 3 == 0: # ADRIAAN: naar 3 gezet voor 11-laags RESNET
                num_maxpools += 1
                self.layers.append(nn.MaxPool2d(pool, pool))

        # Bereken matrix size
        matrix_size = (config.matrixshape[0] // (pool**num_maxpools)) * (
            config.matrixshape[1] // (pool**num_maxpools)
        )
        print(f"Calculated matrix size: {matrix_size}")
        print(f"Calculated flatten size: {matrix_size * config.hidden}")
        
        # Dense layers
        dense_layers = [nn.Flatten(), nn.Linear(matrix_size * config.hidden, config.hidden)]
        
        if config.batch_norm:  # ADRIAAN Added conditional batch norm to dense layers
            dense_layers.append(nn.BatchNorm1d(config.hidden))
        
        dense_layers.append(nn.ReLU())
        
        if config.dropout > 0:  # ADRIAAN Added conditional dropout to dense layers
            dense_layers.append(nn.Dropout(config.dropout))
            
        dense_layers.append(nn.Linear(config.hidden, config.num_classes))
        
        self.dense = nn.Sequential(*dense_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ADRIAAN Added missing forward method
        """Forward pass door het netwerk"""
        # Eerste conv layer
        x = self.first_conv(x)
        x = self.first_relu(x)
        
        # Door alle andere layers (met of zonder skip connections)
        for layer in self.layers:  # ADRIAAN Changed iteration to work with new layer structure
            x = layer(x)
        
        # Door de dense layers
        x = self.dense(x)
        return x