import torch
import torch.nn as nn

class PPNet(nn.Module):
    """
    PP-Net: Non-Contact Blood Pressure Estimation Using Deep Learning With a Novel rPPG Signal
    Based on LRCN (Long-term Recurrent Convolutional Network) architecture.
    """
    def __init__(self, in_channels=3, nb_filters=32, kernel_size=3, frame_depth=20):
        super(PPNet, self).__init__()
        self.in_channels = in_channels
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.frame_depth = frame_depth

        # 2D CNN Feature Extractor (Time-Distributed)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, nb_filters, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(nb_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(nb_filters, nb_filters*2, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(nb_filters*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(nb_filters*2, nb_filters*4, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(nb_filters*4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Recurrent Layer (LSTM)
        # Input size = nb_filters * 4 (128 if nb_filters=32)
        self.lstm = nn.LSTM(input_size=nb_filters*4, hidden_size=128, num_layers=2, batch_first=True, dropout=0.2)

        # Regressor
        # Output: 2 values (SBP, DBP) per video clip or 1 value (BP waveform) per frame?
        # Assuming frame-wise CNIBP signal output for consistency with rPPG framework (B, T, 1)
        self.regressor = nn.Linear(128, 2)

    def forward(self, x):
        # Input x: (B, C, T, H, W)
        b, c, t, h, w = x.shape
        
        # Reshape for TimeDistributed CNN: (B*T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h, w)
        
        # CNN Feature Extraction
        features = self.features(x)  # (B*T, 128, 1, 1)
        features = features.view(b, t, -1)  # (B, T, 128)
        
        # LSTM
        lstm_out, _ = self.lstm(features)  # (B, T, 128)
        
        # Regression
        out = self.regressor(lstm_out)  # (B, T, 1) to match rPPG output shape
        
        return out
