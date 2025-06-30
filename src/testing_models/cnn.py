import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, config):
        super(SimpleCNN, self).__init__()      
        self.num_classes = config.get('output', 5)
        self.dropout_rate = config.get('dropout', 0.3)  
        
        conv1_kernel = config.get('conv1_kernel', 11)
        conv2_kernel = config.get('conv2_kernel', 7) 
        conv3_kernel = config.get('conv3_kernel', 5)
        conv4_kernel = config.get('conv4_kernel', 3)
        
        conv1_channels = config.get('conv1_channels', 32)
        conv2_channels = config.get('conv2_channels', 64)
        conv3_channels = config.get('conv3_channels', 96)
        conv4_channels = config.get('conv4_channels', 128)
        
        fc1_size = config.get('fc1_size', 384)
        fc2_size = config.get('fc2_size', 256)
        fc3_size = config.get('fc3_size', 96)
        
        # CONV LAYERS - zelfde als voor
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=conv1_channels, 
                              kernel_size=conv1_kernel, padding=conv1_kernel//2)
        self.bn1 = nn.BatchNorm1d(conv1_channels)
        
        self.conv2 = nn.Conv1d(in_channels=conv1_channels, out_channels=conv2_channels, 
                              kernel_size=conv2_kernel, padding=conv2_kernel//2)
        self.bn2 = nn.BatchNorm1d(conv2_channels)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(in_channels=conv2_channels, out_channels=conv3_channels, 
                              kernel_size=conv3_kernel, padding=conv3_kernel//2)
        self.bn3 = nn.BatchNorm1d(conv3_channels)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # ← FIX 1: NIEUW!
        
        self.conv4 = nn.Conv1d(in_channels=conv3_channels, out_channels=conv4_channels, 
                              kernel_size=conv4_kernel, padding=conv4_kernel//2)
        self.bn4 = nn.BatchNorm1d(conv4_channels)
        self.avgpool = nn.AdaptiveAvgPool1d(4)  # ← FIX 2: van 8 naar 4
        
        # FC LAYERS - kleinere input
        fc_input_size = conv4_channels * 4  # ← FIX 2: van 8 naar 4
        
        self.fc1 = nn.Linear(fc_input_size, fc1_size)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.dropout2 = nn.Dropout(self.dropout_rate * 0.7)  # ← FIX 3: minder dropout
        
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.dropout3 = nn.Dropout(self.dropout_rate * 0.5)  # ← FIX 3: nog minder
        
        self.fc_final = nn.Linear(fc3_size, self.num_classes)
        
        # FIX 4: Betere weight initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Betere weight initialization voor stabielere training"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            x = x.transpose(1, 2)
            
        # CONV BLOCK 1: 187 punten → 187 punten
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # CONV BLOCK 2: 187 → 93 punten (eerste pooling)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        
        # CONV BLOCK 3: 93 → 46 punten (tweede pooling - NIEUW!)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.maxpool2(x)  # ← FIX 1: extra pooling toegevoegd
        
        # CONV BLOCK 4: 46 → 4 punten (adaptive pooling)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.avgpool(x)
        
        # FLATTEN: [batch, channels*4] - nu 512 in plaats van 1024!
        x = x.view(x.size(0), -1)
        
        # FC LAYERS: met graduele dropout
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)    # Volle dropout (0.3)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)    # Minder dropout (0.21)
        
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout3(x)    # Minste dropout (0.15)
        
        x = self.fc_final(x)
        
        return x
    
    def predict_proba(self, x):
        logits = self.forward(x)
        return F.softmax(logits, dim=1)