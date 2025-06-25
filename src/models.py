import torch
import torch.nn as nn
import torch.nn.functional as F

class ModularCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Base parameters
        self.num_classes = config.get("output", 5)
        self.dropout_rate = config.get("dropout", 0.3)
        self.input_channels = config.get("input_channels", 1)
        # New: flags for skip connections and attention (from config)
        self.use_skip = bool(config.get("use_skip", False))
        self.use_attention = bool(config.get("use_attention", False))
        
        # Architecture parameters
        self.num_conv_layers = config.get("num_conv_layers", 4)
        self.base_channels = config.get("base_channels", 32)
        self.kernel_size = config.get("kernel_size", 3)
        
        # Calculate correct padding to maintain sequence length
        self.padding = (self.kernel_size - 1) // 2  # Voor kernel=5, padding=2
        
        # Convolutional layers definition
        # Conv1 (always present)
        self.conv1 = nn.Conv1d(self.input_channels, self.base_channels, 
                                kernel_size=self.kernel_size, padding=self.padding)
        self.bn1 = nn.BatchNorm1d(self.base_channels)
        # Conv2 (optional, uses double channels)
        self.conv2 = None
        if self.num_conv_layers >= 2:
            self.conv2 = nn.Conv1d(self.base_channels, self.base_channels * 2, 
                                    kernel_size=self.kernel_size, padding=self.padding)
            self.bn2 = nn.BatchNorm1d(self.base_channels * 2)
        # Conv3
        self.conv3 = None
        if self.num_conv_layers >= 3:
            in_c = self.base_channels * (2 ** 1)  # base * 2
            out_c = self.base_channels * (2 ** 2) # base * 4
            self.conv3 = nn.Conv1d(in_c, out_c, kernel_size=self.kernel_size, padding=self.padding)
            self.bn3 = nn.BatchNorm1d(out_c)
        # Conv4
        self.conv4 = None
        if self.num_conv_layers >= 4:
            in_c = self.base_channels * (2 ** 2)  # base * 4
            out_c = self.base_channels * (2 ** 3) # base * 8
            self.conv4 = nn.Conv1d(in_c, out_c, kernel_size=self.kernel_size, padding=self.padding)
            self.bn4 = nn.BatchNorm1d(out_c)
        # Conv5
        self.conv5 = None
        if self.num_conv_layers >= 5:
            in_c = self.base_channels * (2 ** 3)  # base * 8
            out_c = self.base_channels * (2 ** 4) # base * 16
            self.conv5 = nn.Conv1d(in_c, out_c, kernel_size=self.kernel_size, padding=self.padding)
            self.bn5 = nn.BatchNorm1d(out_c)
        # Conv6 (new: allow deeper conv layers)
        self.conv6 = None
        if self.num_conv_layers >= 6:
            in_c = self.base_channels * (2 ** 4)  # base * 16
            out_c = self.base_channels * (2 ** 5) # base * 32
            self.conv6 = nn.Conv1d(in_c, out_c, kernel_size=self.kernel_size, padding=self.padding)
            self.bn6 = nn.BatchNorm1d(out_c)
        # Conv7 (new: allow deeper conv layers)
        self.conv7 = None
        if self.num_conv_layers >= 7:
            in_c = self.base_channels * (2 ** 5)  # base * 32
            out_c = self.base_channels * (2 ** 6) # base * 64
            self.conv7 = nn.Conv1d(in_c, out_c, kernel_size=self.kernel_size, padding=self.padding)
            self.bn7 = nn.BatchNorm1d(out_c)
        
        # Define skip-connection projection conv layers if needed 
        # (1x1 conv to match dimensions for residual addition)
        # These allow the skip path to match the increased channels of the main path.
        if self.use_skip:
            if self.num_conv_layers >= 2:
                self.proj1 = nn.Conv1d(self.base_channels, self.base_channels * 2, kernel_size=1)
            if self.num_conv_layers >= 4:
                # conv3->conv4 skip
                self.proj2 = nn.Conv1d(self.base_channels * 4, self.base_channels * 8, kernel_size=1)
            if self.num_conv_layers >= 6:
                # conv5->conv6 skip
                self.proj3 = nn.Conv1d(self.base_channels * 16, self.base_channels * 32, kernel_size=1)
            # (If num_conv_layers is 7, conv7 has no partner to skip with, so no projection needed for it)
        
        # Pooling layers
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # Global average pooling to collapse time dimension to 1
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected (dense) layers
        final_channels = self.base_channels * (2 ** (self.num_conv_layers - 1))
        self.fc1 = nn.Linear(final_channels, 256)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(self.dropout_rate)
        self.fc_final = nn.Linear(128, self.num_classes)
        
        # Attention layers (Squeeze-and-Excitation block) if enabled
        if self.use_attention:
            # Reduce channels by factor (e.g., 16) then restore, per SE design
            att_hidden = max(1, final_channels // 16)
            self.att_fc1 = nn.Linear(final_channels, att_hidden)
            self.att_fc2 = nn.Linear(att_hidden, final_channels)

    def forward(self, x):
        # Input shape handling
        if x.dim() == 2:
            x = x.unsqueeze(1)    # add channel dim if input is [batch, length]
        elif x.dim() == 3:
            x = x.transpose(1, 2) # ensure shape [batch, channels, length]
        
        # Conv Layer 1 (always present)
        out = F.relu(self.bn1(self.conv1(x)))
        
        # Conv Layer 2
        if self.conv2 is not None:
            # Compute Conv2
            out_conv2 = self.bn2(self.conv2(out))
            if self.use_skip:
                # Skip connection from Conv1 output to Conv2 output
                residual = out  # output from conv1
                # If channels differ, use projection conv to match dimensions
                if hasattr(self, "proj1"):
                    residual = self.proj1(residual)
                # Add skip connection and apply ReLU
                out = F.relu(out_conv2 + residual)
            else:
                out = F.relu(out_conv2)
            out = self.pool(out)  # pool after Conv2 (end of block 1)
        
        # Conv Layer 3
        if self.conv3 is not None:
            out_block_input = out  # input to this block (could be pooled Conv2 output)
            out_conv3 = F.relu(self.bn3(self.conv3(out_block_input)))
            if self.conv4 is not None and self.use_skip:
                # If next layer exists and skip is enabled, prepare for residual block with Conv4
                # (We delay pooling until after Conv4 in this case)
                out_block_inter = out_conv3  # activated output of Conv3
            else:
                # No Conv4 or no skip: just pool after Conv3
                out = self.pool(out_conv3)
                out_block_inter = None  # not used in this scenario
            # Conv Layer 4
            if self.conv4 is not None:
                if self.use_skip:
                    # Compute Conv4 without activation (end of block 2)
                    out_conv4 = self.bn4(self.conv4(out_block_inter))
                    # Skip connection from Conv3 output to Conv4 output
                    residual = out_conv3  # output from conv3 (already ReLU-activated)
                    if hasattr(self, "proj2"):
                        residual = self.proj2(residual)
                    out = F.relu(out_conv4 + residual)
                    out = self.pool(out)  # pool after Conv4 (end of block 2)
                else:
                    out_conv4 = F.relu(self.bn4(self.conv4(out)))  # Conv4 normal path
                    out = self.pool(out_conv4)
        
        # Conv Layer 5
        if self.conv5 is not None:
            out_block_input = out  # input to block 3 (could be pooled Conv4 or Conv3 output)
            out_conv5 = F.relu(self.bn5(self.conv5(out_block_input)))
            if self.conv6 is not None and self.use_skip:
                # Prepare for residual block with Conv6
                out_block_inter = out_conv5
            else:
                out = self.pool(out_conv5)
                out_block_inter = None
            # Conv Layer 6
            if self.conv6 is not None:
                if self.use_skip:
                    out_conv6 = self.bn6(self.conv6(out_block_inter))
                    # Skip connection from Conv5 output to Conv6 output
                    residual = out_conv5
                    if hasattr(self, "proj3"):
                        residual = self.proj3(residual)
                    out = F.relu(out_conv6 + residual)
                    out = self.pool(out)  # pool after Conv6 (end of block 3)
                else:
                    out_conv6 = F.relu(self.bn6(self.conv6(out)))
                    out = self.pool(out_conv6)
        
        # Conv Layer 7 (if any, stands alone if present)
        if self.conv7 is not None:
            out_conv7 = F.relu(self.bn7(self.conv7(out)))
            out = self.pool(out_conv7)
        
        # Attention mechanism (Squeeze-and-Excitation) if enabled
        if self.use_attention:
            # Squeeze: global average pooling to get channel-wise stats
            se = out.mean(dim=2)  # shape [batch, channels]
            # Excitation: two small FC layers to generate attention weights
            se = F.relu(self.att_fc1(se))
            se = torch.sigmoid(self.att_fc2(se))
            se = se.unsqueeze(-1)  # shape [batch, channels, 1]
            # Scale: multiply weights with feature map to reweight channel importance
            out = out * se
        
        # Global average pool to ensure output shape [batch, channels, 1]
        out = self.global_pool(out)  # output shape becomes [batch, channels, 1]
        out = out.squeeze(-1)        # shape [batch, channels]
        
        # Fully connected classification layers
        out = F.relu(self.fc1(out))
        out = self.dropout1(out)
        out = F.relu(self.fc2(out))
        out = self.dropout2(out)
        out = self.fc_final(out)   # logits for each class
        
        return out

    def predict_proba(self, x):
        """Probability output for each class."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)