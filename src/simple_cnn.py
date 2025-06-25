import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    Simpele CNN volgens de oorspronkelijke architectuur, maar met slimmere hyperparameter keuzes:
    
    INPUT
    Convolution -> Batch Norm -> Relu
    Convolution -> Batch Norm -> Relu -> Max-Pool
    Convolution -> Batch Norm -> Relu  
    Convolution -> Batch Norm -> Relu -> Avg-Pool
    FC -> Relu -> FC -> Relu -> FC -> Relu -> FC -> Softmax
    """
    
    def __init__(self, config):
        super(SimpleCNN, self).__init__()
        
        # ============================================================
        # SLIMMERE HYPERPARAMETER KEUZES
        # ============================================================
        
        self.num_classes = config.get('output', 5)
        self.dropout_rate = config.get('dropout', 0.3)  # Verhoogd van 0.1 naar 0.3 voor betere regularization
        
        # ============================================================
        # CONVOLUTION LAYERS - SLIMMERE KERNEL GROOTTES
        # ============================================================
        
        # Conv1: Grotere kernel (11 i.p.v. 7) om langere hartslag patronen te vangen
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=11, padding=5)
        self.bn1 = nn.BatchNorm1d(32)
        
        # Conv2: Kernel 7 (i.p.v. 5) voor middelgrote patronen + MaxPool
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # Max pooling voor scherpe features
        
        # Conv3: Meer channels (128 -> 96) voor betere balans
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=96, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(96)
        
        # Conv4: Kleinere laatste layer (256 -> 128) om overfitting te voorkomen
        self.conv4 = nn.Conv1d(in_channels=96, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.avgpool = nn.AdaptiveAvgPool1d(8)  # Behoud 8 voor global context
        
        # ============================================================
        # FULLY CONNECTED LAYERS - SLIMMERE GROOTTES
        # ============================================================
        
        # Input size: 128 channels * 8 positions = 1024 (kleiner dan voorheen)
        fc_input_size = 128 * 8
        
        # FC1: Kleiner (512 -> 384) om overfitting te voorkomen
        self.fc1 = nn.Linear(fc_input_size, 384)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        
        # FC2: Behoud 256, maar met meer dropout
        self.fc2 = nn.Linear(384, 256)
        self.dropout2 = nn.Dropout(self.dropout_rate)
        
        # FC3: Kleiner (128 -> 96) voor betere generalization
        self.fc3 = nn.Linear(256, 96)
        self.dropout3 = nn.Dropout(self.dropout_rate)
        
        # Final FC: 96 -> 5 classes
        self.fc_final = nn.Linear(96, self.num_classes)
    
    def forward(self, x):
        """
        Forward pass - zelfde structuur als origineel
        """
        
        # Input shape aanpassen voor Conv1D: (batch, channels, length)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, seq_len) -> (batch, 1, seq_len)
        elif x.dim() == 3:
            x = x.transpose(1, 2)  # (batch, seq_len, 1) -> (batch, 1, seq_len)
            
        # ============================================================
        # CONVOLUTION PART - zelfde structuur, betere parameters
        # ============================================================
        
        # Conv blok 1: Convolution -> Batch Norm -> ReLU
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Conv blok 2: Convolution -> Batch Norm -> ReLU -> Max-Pool
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool1(x)  # Max pooling voor feature detection
        
        # Conv blok 3: Convolution -> Batch Norm -> ReLU
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Conv blok 4: Convolution -> Batch Norm -> ReLU -> Avg-Pool
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.avgpool(x)  # Average pooling voor global context
        
        # ============================================================
        # FULLY CONNECTED PART - zelfde structuur, betere groottes
        # ============================================================
        
        # Flatten voor fully connected layers
        x = x.view(x.size(0), -1)  # (batch, 128*8) = (batch, 1024)
        
        # FC layer 1: Fully connected -> ReLU -> Dropout
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # FC layer 2: Fully connected -> ReLU -> Dropout
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # FC layer 3: Fully connected -> ReLU -> Dropout
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Final FC: Fully connected (geen ReLU, geen dropout)
        x = self.fc_final(x)
        
        return x
    
    def predict_proba(self, x):
        """Voorspel class probabilities (met softmax)"""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

