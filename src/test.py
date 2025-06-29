import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexibleCNN(nn.Module):
    """
    Flexibele CNN die exact jouw beste Ray Tune configuratie kan gebruiken.
    Gebaseerd op jouw best config parameters van 25 juni 2025.
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Basis parameters
        self.input_channels = config.get("input_channels", 1)
        self.num_classes = config.get("output", 5)
        self.dropout_rate = config.get("dropout", 0.3)
        self.num_conv_layers = config.get("num_conv_layers", 4)
        
        # Conv Layer 0: 1 -> 64, kernel=7, MaxPool
        self.conv0 = nn.Conv1d(
            self.input_channels,
            config.get("layer_0_out_channels", 64),
            config.get("layer_0_kernel_size", 7),
            padding=config.get("layer_0_kernel_size", 7) // 2
        )
        self.bn0 = nn.BatchNorm1d(config.get("layer_0_out_channels", 64))
        self.pool0 = self._get_pool_layer(config.get("layer_0_pool", "max"))
        
        # Conv Layer 1: 64 -> 64, kernel=7, MaxPool
        self.conv1 = nn.Conv1d(
            config.get("layer_0_out_channels", 64),
            config.get("layer_1_out_channels", 64),
            config.get("layer_1_kernel_size", 7),
            padding=config.get("layer_1_kernel_size", 7) // 2
        )
        self.bn1 = nn.BatchNorm1d(config.get("layer_1_out_channels", 64))
        self.pool1 = self._get_pool_layer(config.get("layer_1_pool", "max"))
        
        # Conv Layer 2: 64 -> 96, kernel=3, AvgPool
        self.conv2 = nn.Conv1d(
            config.get("layer_1_out_channels", 64),
            config.get("layer_2_out_channels", 96),
            config.get("layer_2_kernel_size", 3),
            padding=config.get("layer_2_kernel_size", 3) // 2
        )
        self.bn2 = nn.BatchNorm1d(config.get("layer_2_out_channels", 96))
        self.pool2 = self._get_pool_layer(config.get("layer_2_pool", "avg"))
        
        # Conv Layer 3: 96 -> 160, kernel=3, No Pool
        self.conv3 = nn.Conv1d(
            config.get("layer_2_out_channels", 96),
            config.get("layer_3_out_channels", 160),
            config.get("layer_3_kernel_size", 3),
            padding=config.get("layer_3_kernel_size", 3) // 2
        )
        self.bn3 = nn.BatchNorm1d(config.get("layer_3_out_channels", 160))
        self.pool3 = self._get_pool_layer(config.get("layer_3_pool", "none"))
        
        # Optionele extra conv layers (als num_conv_layers > 4)
        self.extra_layers = nn.ModuleList()
        if self.num_conv_layers > 4:
            for i in range(4, self.num_conv_layers):
                # Voor extra layers, gebruik default waarden
                in_channels = config.get(f"layer_{i-1}_out_channels", 160)
                out_channels = config.get(f"layer_{i}_out_channels", 160)
                kernel_size = config.get(f"layer_{i}_kernel_size", 3)
                
                conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               padding=kernel_size // 2)
                bn = nn.BatchNorm1d(out_channels)
                pool = self._get_pool_layer(config.get(f"layer_{i}_pool", "max"))
                
                self.extra_layers.append(nn.ModuleDict({
                    'conv': conv,
                    'bn': bn,
                    'pool': pool
                }))
        
        # Global adaptive pooling om dimensie problemen te voorkomen
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # FC Layers met configureerbare sizes
        final_conv_channels = config.get(f"layer_{self.num_conv_layers-1}_out_channels", 160)
        
        self.fc1 = nn.Linear(final_conv_channels, config.get("fc1_size", 256))
        self.dropout1 = nn.Dropout(self.dropout_rate)
        
        self.fc2 = nn.Linear(config.get("fc1_size", 256), config.get("fc2_size", 256))
        self.dropout2 = nn.Dropout(self.dropout_rate)
        
        self.fc3 = nn.Linear(config.get("fc2_size", 256), config.get("fc3_size", 96))
        self.dropout3 = nn.Dropout(self.dropout_rate)
        
        self.output_layer = nn.Linear(config.get("fc3_size", 96), self.num_classes)
        
        # Store config voor debugging
        self.config = config
        
    def _get_pool_layer(self, pool_type):
        """Helper functie om pooling layers te maken"""
        if pool_type == "max":
            return nn.MaxPool1d(kernel_size=2, stride=2)
        elif pool_type == "avg":
            return nn.AvgPool1d(kernel_size=2, stride=2)
        elif pool_type == "none" or pool_type is None:
            return nn.Identity()  # Geen pooling
        else:
            return nn.MaxPool1d(kernel_size=2, stride=2)  # Default naar max
    
    def forward(self, x):
        # Input preprocessing - zorg voor juiste dimensies
        if x.dim() == 2:  # [batch, features]
            x = x.unsqueeze(1)  # [batch, 1, features]
        elif x.dim() == 3 and x.shape[1] != self.input_channels:
            x = x.transpose(1, 2)  # [batch, features, channels] -> [batch, channels, features]
        
        # Conv Layer 0
        x = F.relu(self.bn0(self.conv0(x)))
        if self.pool0 is not None:
            x = self.pool0(x)
        
        # Conv Layer 1
        x = F.relu(self.bn1(self.conv1(x)))
        if self.pool1 is not None:
            x = self.pool1(x)
        
        # Conv Layer 2
        x = F.relu(self.bn2(self.conv2(x)))
        if self.pool2 is not None:
            x = self.pool2(x)
        
        # Conv Layer 3
        x = F.relu(self.bn3(self.conv3(x)))
        if self.pool3 is not None and not isinstance(self.pool3, nn.Identity):
            x = self.pool3(x)
        
        # Extra conv layers (als aanwezig)
        for layer_dict in self.extra_layers:
            x = F.relu(layer_dict['bn'](layer_dict['conv'](x)))
            if not isinstance(layer_dict['pool'], nn.Identity):
                x = layer_dict['pool'](x)
        
        # Global pooling en flatten
        x = self.global_pool(x)  # [batch, channels, 1]
        x = x.squeeze(-1)        # [batch, channels]
        
        # FC Layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        
        x = self.output_layer(x)
        
        return x
    
    def predict_proba(self, x):
        """Compatibility functie voor mltrainer"""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def print_architecture(self):
        """Print model architectuur voor debugging"""
        print(f"FlexibleCNN Architecture:")
        print(f"Input channels: {self.input_channels}")
        print(f"Number of conv layers: {self.num_conv_layers}")
        print(f"Dropout rate: {self.dropout_rate}")
        print()
        
        for i in range(min(4, self.num_conv_layers)):
            layer_name = f"layer_{i}"
            in_ch = self.config.get(f"layer_{i-1}_out_channels", self.input_channels) if i > 0 else self.input_channels
            out_ch = self.config.get(f"layer_{i}_out_channels", 64)
            kernel = self.config.get(f"layer_{i}_kernel_size", 3)
            pool = self.config.get(f"layer_{i}_pool", "max")
            print(f"Conv{i}: {in_ch} -> {out_ch}, kernel={kernel}, pool={pool}")
        
        print()
        print(f"FC1: {self.config.get('fc1_size', 256)}")
        print(f"FC2: {self.config.get('fc2_size', 256)}")
        print(f"FC3: {self.config.get('fc3_size', 96)}")
        print(f"Output: {self.num_classes}")


# Jouw exacte best config voor gemakkelijk testen
BEST_CONFIG = {
    "attention": False,
    "batch_size": 32,
    "data_dir": "/path/to/data",  # Pas aan naar jouw pad
    "dataset_name": "heart_big",
    "dropout": 0.4195476887814652,
    "fc1_size": 256,
    "fc2_size": 256,
    "fc3_size": 96,
    "input_channels": 1,
    "layer_0_kernel_size": 7,
    "layer_0_out_channels": 64,
    "layer_0_pool": "max",
    "layer_1_kernel_size": 7,
    "layer_1_out_channels": 64,
    "layer_1_pool": "max",
    "layer_2_kernel_size": 3,
    "layer_2_out_channels": 96,
    "layer_2_pool": "avg",
    "layer_3_kernel_size": 3,
    "layer_3_out_channels": 160,
    "layer_3_pool": "none",
    "num_conv_layers": 4,
    "output": 5,
    "skip_layers": [],
    "squeeze_excite": False,
    "target_count": 15000
}

# Test functie
if __name__ == "__main__":
    # Maak model met jouw beste config
    model = FlexibleCNN(BEST_CONFIG)
    
    # Print architectuur
    model.print_architecture()
    
    # Test met dummy data (187 features zoals in jouw dataset)
    dummy_input = torch.randn(32, 187)  # batch_size=32, features=187
    
    print(f"\nInput shape: {dummy_input.shape}")
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Expected: [32, 5] for 5 classes")
    
    # Test predict_proba
    probs = model.predict_proba(dummy_input)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Probabilities sum to 1: {torch.allclose(probs.sum(dim=1), torch.ones(32))}")