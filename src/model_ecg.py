from typing import Optional

import toml
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_config(config_path: str = "config.toml") -> dict:
    """
    Load configuration from a TOML file.
    """
    try:
        return toml.load(config_path)
    except FileNotFoundError:
        print(f"Config file {config_path} not found. Using default values.")
        return get_default_config()
    except Exception as e:
        print(f"Error loading config: {e}. Using default values.")
        return get_default_config()


def get_default_config() -> dict:
    """
    Return default configuration values if TOML file is not available.
    """
    return {
        'model': {
            'output_classes': 5,
            'dropout_rate': 0.3,
            'se_reduction': 16
        },
        'conv_layers': {
            'conv1': {'out_channels': 32, 'kernel_size': 11, 'padding': 5},
            'conv2': {'out_channels': 64, 'kernel_size': 7, 'padding': 3},
            'conv3': {'out_channels': 160, 'kernel_size': 3, 'padding': 1}
        },
        'residual_blocks': {
            'resblock1': {'out_channels': 96, 'kernel_size': 5, 'padding': 2},
            'resblock2': {'out_channels': 128, 'kernel_size': 3, 'padding': 1}
        },
        'pooling': {
            'maxpool_kernel': 2,
            'maxpool_stride': 2,
            'avgpool_output': 8
        },
        'fully_connected': {
            'fc1_size': 384,
            'fc2_size': 256,
            'fc3_size': 96
        }
    }


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block to recalibrate channel-wise feature responses.
    """
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.squeeze_fc1 = nn.Linear(channels, channels // reduction)
        self.squeeze_fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = F.relu(self.squeeze_fc1(y))
        y = torch.sigmoid(self.squeeze_fc2(y)).view(b, c, 1)
        return x * y


class ResidualBlock(nn.Module):
    """
    Residual block with optional channel matching for skip connections.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int) -> None:
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.match_channels: Optional[nn.Conv1d] = None
        if in_channels != out_channels:
            self.match_channels = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv(x)
        out = self.bn(out)
        if self.match_channels:
            identity = self.match_channels(identity)
        out += identity
        out = self.relu(out)
        return out


class SimpleCNN(nn.Module):
    """
    A custom 1D CNN model with SE blocks and residual connections.
    Now configurable via TOML file.
    """
    def __init__(self, config_path: str = "config.toml") -> None:
        super(SimpleCNN, self).__init__()
        
        # Load configuration
        config = load_config(config_path)
        
        # Extract configuration values
        model_config = config['model']
        conv_config = config['conv_layers']
        res_config = config['residual_blocks']
        pool_config = config['pooling']
        fc_config = config['fully_connected']
        
        self.num_classes = model_config['output_classes']
        self.dropout_rate = model_config['dropout_rate']
        se_reduction = model_config['se_reduction']

        # First convolutional layer
        conv1_cfg = conv_config['conv1']
        self.conv1 = nn.Conv1d(
            in_channels=1, 
            out_channels=conv1_cfg['out_channels'], 
            kernel_size=conv1_cfg['kernel_size'], 
            padding=conv1_cfg['padding']
        )
        self.bn1 = nn.BatchNorm1d(conv1_cfg['out_channels'])

        # Second convolutional layer
        conv2_cfg = conv_config['conv2']
        self.conv2 = nn.Conv1d(
            in_channels=conv1_cfg['out_channels'], 
            out_channels=conv2_cfg['out_channels'], 
            kernel_size=conv2_cfg['kernel_size'], 
            padding=conv2_cfg['padding']
        )
        self.bn2 = nn.BatchNorm1d(conv2_cfg['out_channels'])
        self.se2 = SEBlock(conv2_cfg['out_channels'], se_reduction)
        
        # Max pooling
        self.maxpool1 = nn.MaxPool1d(
            kernel_size=pool_config['maxpool_kernel'], 
            stride=pool_config['maxpool_stride']
        )

        # First residual block
        res1_cfg = res_config['resblock1']
        self.resblock1 = ResidualBlock(
            conv2_cfg['out_channels'], 
            res1_cfg['out_channels'], 
            kernel_size=res1_cfg['kernel_size'], 
            padding=res1_cfg['padding']
        )
        self.se3 = SEBlock(res1_cfg['out_channels'], se_reduction)

        # Second residual block
        res2_cfg = res_config['resblock2']
        self.resblock2 = ResidualBlock(
            res1_cfg['out_channels'], 
            res2_cfg['out_channels'], 
            kernel_size=res2_cfg['kernel_size'], 
            padding=res2_cfg['padding']
        )
        self.se4 = SEBlock(res2_cfg['out_channels'], se_reduction)

        # Third convolutional layer
        conv3_cfg = conv_config['conv3']
        self.conv3 = nn.Conv1d(
            in_channels=res2_cfg['out_channels'], 
            out_channels=conv3_cfg['out_channels'], 
            kernel_size=conv3_cfg['kernel_size'], 
            padding=conv3_cfg['padding']
        )
        self.bn3 = nn.BatchNorm1d(conv3_cfg['out_channels'])
        self.se5 = SEBlock(conv3_cfg['out_channels'], se_reduction)

        # Average pooling
        self.avgpool = nn.AdaptiveAvgPool1d(pool_config['avgpool_output'])
        fc_input_size = conv3_cfg['out_channels'] * pool_config['avgpool_output']

        # Fully connected layers
        self.fc1 = nn.Linear(fc_input_size, fc_config['fc1_size'])
        self.dropout1 = nn.Dropout(self.dropout_rate)

        self.fc2 = nn.Linear(fc_config['fc1_size'], fc_config['fc2_size'])
        self.dropout2 = nn.Dropout(self.dropout_rate)

        self.fc3 = nn.Linear(fc_config['fc2_size'], fc_config['fc3_size'])
        self.dropout3 = nn.Dropout(self.dropout_rate)

        self.fc_final = nn.Linear(fc_config['fc3_size'], self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            x = x.transpose(1, 2)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.se2(x)
        x = self.maxpool1(x)

        x = self.resblock1(x)
        x = self.se3(x)

        x = self.resblock2(x)
        x = self.se4(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.se5(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout3(x)

        x = self.fc_final(x)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities using softmax.
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)