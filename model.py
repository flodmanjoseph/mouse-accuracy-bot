"""Fully-convolutional CNN for target heatmap prediction."""
import torch
import torch.nn as nn


class TargetDetector(nn.Module):
    """Simple fully-convolutional network that outputs a spatial heatmap.

    Input:  (B, 3, 180, 320)  — RGB screenshot
    Output: (B, 1, 45, 80)    — heatmap where bright = target center

    Architecture (~200K parameters):
        Conv(3→32)  → BN → ReLU → MaxPool(2)   # (32, 90, 160)
        Conv(32→64) → BN → ReLU → MaxPool(2)   # (64, 45, 80)
        Conv(64→64) → BN → ReLU                 # (64, 45, 80)
        Conv(64→32) → BN → ReLU                 # (32, 45, 80)
        Conv(32→1)  → Sigmoid                   # (1, 45, 80)
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: downsample 2x
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: downsample 2x
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: refine
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 4: reduce channels
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Head: single channel heatmap
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.features(x)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = TargetDetector()
    print(f"Parameters: {count_parameters(model):,}")

    # Test with dummy input
    dummy = torch.randn(1, 3, 180, 320)
    out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
    print(f"Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
