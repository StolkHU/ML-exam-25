import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.squeeze_fc1 = nn.Linear(channels, channels // reduction)
        self.squeeze_fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = F.relu(self.squeeze_fc1(y))
        y = torch.sigmoid(self.squeeze_fc2(y)).view(b, c, 1)
        return x * y


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.match_channels = None
        if in_channels != out_channels:
            self.match_channels = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = x
        out = self.conv(x)
        out = self.bn(out)
        if self.match_channels:
            identity = self.match_channels(identity)
        out += identity
        out = self.relu(out)
        return out


class SimpleCNN(nn.Module):
    def __init__(self, config):
        super(SimpleCNN, self).__init__()
        self.num_classes = config.get('output', 5)
        self.dropout_rate = config.get('dropout', 0.3)

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=11, padding=5)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.se2 = SEBlock(64)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.resblock1 = ResidualBlock(64, 96, kernel_size=5, padding=2)
        self.se3 = SEBlock(96)

        self.resblock2 = ResidualBlock(96, 128, kernel_size=3, padding=1)
        self.se4 = SEBlock(128)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=160, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(160)
        self.se5 = SEBlock(160)

        self.avgpool = nn.AdaptiveAvgPool1d(8)
        fc_input_size = 160 * 8

        self.fc1 = nn.Linear(fc_input_size, 384)
        self.dropout1 = nn.Dropout(self.dropout_rate)

        self.fc2 = nn.Linear(384, 256)
        self.dropout2 = nn.Dropout(self.dropout_rate)

        self.fc3 = nn.Linear(256, 96)
        self.dropout3 = nn.Dropout(self.dropout_rate)

        self.fc_final = nn.Linear(96, self.num_classes)

    def forward(self, x):
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

    def predict_proba(self, x):
        logits = self.forward(x)
        return F.softmax(logits, dim=1)