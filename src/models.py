import torch
import torch.nn as nn
import torch.nn.functional as F

class ModularCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        # AS: Basis parameters
        self.num_classes = config.get("output", 5)
        self.dropout_rate = config.get("dropout", 0.3)
        self.input_channels = config.get("input_channels", 1)
        
        # AS: Architectuur parameters
        self.num_conv_layers = config.get("num_conv_layers", 4)
        self.base_channels = config.get("base_channels", 32)
        self.kernel_size = config.get("kernel_size", 3)

        # AS: Simpele conv layers - vaste architectuur
        self.conv1 = nn.Conv1d(self.input_channels, self.base_channels, kernel_size=self.kernel_size, padding=1)
        self.bn1 = nn.BatchNorm1d(self.base_channels)
        
        self.conv2 = nn.Conv1d(self.base_channels, self.base_channels * 2, kernel_size=self.kernel_size, padding=1)
        self.bn2 = nn.BatchNorm1d(self.base_channels * 2)
        
        self.conv3 = nn.Conv1d(self.base_channels * 2, self.base_channels * 4, kernel_size=self.kernel_size, padding=1)
        self.bn3 = nn.BatchNorm1d(self.base_channels * 4)
        
        # AS: Optionele 4e layer
        if self.num_conv_layers >= 4:
            self.conv4 = nn.Conv1d(self.base_channels * 4, self.base_channels * 8, kernel_size=self.kernel_size, padding=1)
            self.bn4 = nn.BatchNorm1d(self.base_channels * 8)
        else:
            self.conv4 = None
            self.bn4 = None
        
        # AS: Optionele 5e layer
        if self.num_conv_layers >= 5:
            self.conv5 = nn.Conv1d(self.base_channels * 8, self.base_channels * 16, kernel_size=self.kernel_size, padding=1)
            self.bn5 = nn.BatchNorm1d(self.base_channels * 16)
        else:
            self.conv5 = None
            self.bn5 = None

        # AS: Pooling layers
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # AS: Global average pooling om dimensie problemen te voorkomen
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # AS: Vaste FC layers - geen berekeningen nodig
        final_channels = self.base_channels * (2 ** (self.num_conv_layers - 1))
        
        self.fc1 = nn.Linear(final_channels, 256)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(self.dropout_rate)
        self.fc_final = nn.Linear(128, self.num_classes)

    def forward(self, x):
        # AS: Input preprocessing
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            x = x.transpose(1, 2)

        # AS: Conv layer 1 (altijd)
        x = F.relu(self.bn1(self.conv1(x)))
        
        # AS: Conv layer 2 (altijd)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        # AS: Conv layer 3 (altijd)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # AS: Conv layer 4 (optioneel)
        if self.conv4 is not None:
            x = F.relu(self.bn4(self.conv4(x)))
            x = self.pool(x)
        
        # AS: Conv layer 5 (optioneel)
        if self.conv5 is not None:
            x = F.relu(self.bn5(self.conv5(x)))
            x = self.pool(x)
        
        # AS: Global pooling - altijd 1D output ongeacht input
        x = self.global_pool(x)  # Shape: [batch, channels, 1]
        x = x.squeeze(-1)        # Shape: [batch, channels]
        
        # AS: FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc_final(x)
        
        return x

    def predict_proba(self, x):
        """AS: Compatibiliteit"""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)