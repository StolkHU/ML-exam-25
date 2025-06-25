import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExciteBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        batch, channels, length = x.size()
        y = F.adaptive_avg_pool1d(x, 1).view(batch, channels)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(batch, channels, 1)
        return x * y.expand_as(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv1d(channels, channels, kernel_size=1)
        self.key = nn.Conv1d(channels, channels, kernel_size=1)
        self.value = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn_weights = torch.softmax(torch.bmm(Q.transpose(1, 2), K), dim=-1)
        attn_output = torch.bmm(attn_weights, V.transpose(1, 2)).transpose(1, 2)
        return attn_output

class ModularCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config.get("output", 5)
        self.dropout_rate = config.get("dropout", 0.3)
        self.use_se = config.get("squeeze_excite", False)
        self.use_attention = config.get("attention", False)
        self.skip_layers = config.get("skip_layers", [])
        self.input_channels = config.get("input_channels", 1)
        self.num_conv_layers = config.get("num_conv_layers", 4)
        self.min_out_channels = config.get("min_out_channels", 32)
        self.max_out_channels = config.get("max_out_channels", 128)
        self.kernel_sizes = config.get("kernel_sizes", [3, 5, 7])
        self.pool_strategy = config.get("pool_strategy", "none")

        self.conv_blocks = nn.ModuleList()
        in_channels = self.input_channels

        for i in range(self.num_conv_layers):
            if i in self.skip_layers:
                continue
            out_channels = int(self.min_out_channels + (self.max_out_channels - self.min_out_channels) * i / max(1, self.num_conv_layers - 1))
            kernel_size = self.kernel_sizes[i % len(self.kernel_sizes)]
            padding = kernel_size // 2

            conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
            bn = nn.BatchNorm1d(out_channels)
            se = SqueezeExciteBlock(out_channels) if self.use_se else None
            attn = AttentionBlock(out_channels) if self.use_attention else None

            if self.pool_strategy == "max":
                pool = nn.MaxPool1d(kernel_size=2, stride=2)
            elif self.pool_strategy == "avg":
                pool = nn.AdaptiveAvgPool1d(8)
            else:
                pool = None

            block = nn.ModuleDict({
                "conv": conv,
                "bn": bn,
                "se": se,
                "attn": attn,
                "pool": pool
            })

            self.conv_blocks.append(block)
            in_channels = out_channels

        self.flatten_dim = in_channels * 8 if self.pool_strategy == "avg" else None
        self.fc1_size = config.get("fc1_size", 384)
        self.fc2_size = config.get("fc2_size", 256)
        self.fc3_size = config.get("fc3_size", 96)

        self.fc1 = nn.Linear(self.flatten_dim, self.fc1_size)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(self.fc1_size, self.fc2_size)
        self.dropout2 = nn.Dropout(self.dropout_rate)
        self.fc3 = nn.Linear(self.fc2_size, self.fc3_size)
        self.dropout3 = nn.Dropout(self.dropout_rate)
        self.fc_final = nn.Linear(self.fc3_size, self.num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            x = x.transpose(1, 2)

        for block in self.conv_blocks:
            x = block["conv"](x)
            x = block["bn"](x)
            x = F.relu(x)
            if block["se"]:
                x = block["se"](x)
            if block["attn"]:
                x = block["attn"](x)
            if block["pool"]:
                x = block["pool"](x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc_final(x)
        return x

    def predict_proba(self, x):
        logits = self.forward(x)
        return F.softmax(logits, dim=1)