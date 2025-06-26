import torch
import torch.nn as nn
import torch.nn.functional as F

class ModularCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config.get("output", 5)
        self.dropout_rate = config.get("dropout", 0.3)
        self.input_channels = config.get("input_channels", 1)
        self.use_skip = bool(config.get("use_skip", False))
        self.use_attention = bool(config.get("use_attention", False))

        self.num_conv_layers = config.get("num_conv_layers", 4)
        self.base_channels = config.get("base_channels", 32)
        self.kernel_size = config.get("kernel_size", 3)
        self.padding = (self.kernel_size - 1) // 2

        # FC layer sizes
        fc1_size = config.get("fc1_size", 256)
        fc2_size = config.get("fc2_size", 128)
        fc3_size = config.get("fc3_size", 64)  # Optional

        # Convolutional layers
        self.conv1 = nn.Conv1d(self.input_channels, self.base_channels, self.kernel_size, padding=self.padding)
        self.bn1 = nn.BatchNorm1d(self.base_channels)

        self.conv2 = self.bn2 = None
        if self.num_conv_layers >= 2:
            self.conv2 = nn.Conv1d(self.base_channels, self.base_channels * 2, self.kernel_size, padding=self.padding)
            self.bn2 = nn.BatchNorm1d(self.base_channels * 2)

        self.conv3 = self.bn3 = None
        if self.num_conv_layers >= 3:
            self.conv3 = nn.Conv1d(self.base_channels * 2, self.base_channels * 4, self.kernel_size, padding=self.padding)
            self.bn3 = nn.BatchNorm1d(self.base_channels * 4)

        self.conv4 = self.bn4 = None
        if self.num_conv_layers >= 4:
            self.conv4 = nn.Conv1d(self.base_channels * 4, self.base_channels * 8, self.kernel_size, padding=self.padding)
            self.bn4 = nn.BatchNorm1d(self.base_channels * 8)

        self.conv5 = self.bn5 = None
        if self.num_conv_layers >= 5:
            self.conv5 = nn.Conv1d(self.base_channels * 8, self.base_channels * 16, self.kernel_size, padding=self.padding)
            self.bn5 = nn.BatchNorm1d(self.base_channels * 16)

        # Skip projections
        if self.use_skip:
            if self.num_conv_layers >= 2:
                self.proj1 = nn.Conv1d(self.base_channels, self.base_channels * 2, kernel_size=1)
            if self.num_conv_layers >= 4:
                self.proj2 = nn.Conv1d(self.base_channels * 4, self.base_channels * 8, kernel_size=1)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        final_channels = self.base_channels * (2 ** (self.num_conv_layers - 1))

        self.fc1 = nn.Linear(final_channels, fc1_size)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.dropout2 = nn.Dropout(self.dropout_rate)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.dropout3 = nn.Dropout(self.dropout_rate)
        self.fc_final = nn.Linear(fc3_size, self.num_classes)

        if self.use_attention:
            att_hidden = max(1, final_channels // 16)
            self.att_fc1 = nn.Linear(final_channels, att_hidden)
            self.att_fc2 = nn.Linear(att_hidden, final_channels)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            x = x.transpose(1, 2)

        out = F.relu(self.bn1(self.conv1(x)))

        if self.conv2:
            out_conv2 = self.bn2(self.conv2(out))
            if self.use_skip and hasattr(self, "proj1"):
                out = F.relu(out_conv2 + self.proj1(out))
            else:
                out = F.relu(out_conv2)
            out = self.pool(out)

        if self.conv3:
            out_conv3 = F.relu(self.bn3(self.conv3(out)))
            out = out_conv3

        if self.conv4:
            out_conv4 = self.bn4(self.conv4(out))
            if self.use_skip and hasattr(self, "proj2"):
                out = F.relu(out_conv4 + self.proj2(out))
            else:
                out = F.relu(out_conv4)
            out = self.pool(out)

        if self.conv5:
            out = F.relu(self.bn5(self.conv5(out)))
            out = self.pool(out)

        if self.use_attention:
            se = out.mean(dim=2)
            se = F.relu(self.att_fc1(se))
            se = torch.sigmoid(self.att_fc2(se)).unsqueeze(-1)
            out = out * se

        out = self.global_pool(out).squeeze(-1)

        out = F.relu(self.fc1(out))
        out = self.dropout1(out)
        out = F.relu(self.fc2(out))
        out = self.dropout2(out)
        out = F.relu(self.fc3(out))
        out = self.dropout3(out)
        out = self.fc_final(out)
        return out

    def predict_proba(self, x):
        logits = self.forward(x)
        return F.softmax(logits, dim=1)