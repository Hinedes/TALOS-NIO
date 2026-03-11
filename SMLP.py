import torch
import torch.nn as nn

class SpectralMLP(nn.Module):
    """
    Fully stateless MLP for TALOS NIO.
    Accepts precomputed, flattened spectral magnitudes.
    Spectral Dropout applied immediately at the input to prevent memorization.
    """
    def __init__(self, input_dim=198):
        super().__init__()
        
        # 1. Blind the network to raw frequency bins immediately
        self.spectral_dropout = nn.Dropout(p=0.4)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),  
            nn.ReLU(),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # 3-DOF Translation (Displacement / Velocity)
        self.head_trans = nn.Linear(64, 3)
        
        # 4-DOF Quaternion Orientation
        self.head_quat = nn.Linear(64, 4)
        
        # 3-DOF Aleatoric Uncertainty (Log-Variance for X, Y, Z)
        self.head_cov = nn.Linear(64, 3)
        
        # CRITICAL: Initialize covariance head to zero to prevent early gradient explosion
        # This starts the network with a predicted variance of exp(0) = 1.0
        nn.init.zeros_(self.head_cov.weight)
        nn.init.zeros_(self.head_cov.bias)

    def forward(self, x):
        # Apply dropout to the raw frequency bins BEFORE they enter the MLP
        x_dropped = self.spectral_dropout(x)
        
        features = self.encoder(x_dropped)
        
        translation = self.head_trans(features)
        quaternion = self.head_quat(features)
        log_var = self.head_cov(features)
        
        # CRITICAL: Restored your quaternion normalization to prevent space warping
        quaternion = nn.functional.normalize(quaternion, p=2, dim=1)
        
        return translation, quaternion, log_var

# --- System Check ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Nymeria uses 64-sample windows (33 bins * 6 channels = 198)
    model = SpectralMLP(input_dim=198).to(device)

    mock_input = torch.randn(64, 198).to(device)

    pred_trans, pred_quat, pred_cov = model(mock_input)
    print(":: Network Output Diagnostics ::")
    print(f"Predicted Translation Shape:   {pred_trans.shape}") 
    print(f"Predicted Quaternion Shape:    {pred_quat.shape}")
    print(f"Predicted Log-Variance Shape:  {pred_cov.shape}")