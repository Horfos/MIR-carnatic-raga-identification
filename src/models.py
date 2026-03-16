import torch
import torch.nn as nn


class BaselineCNN(nn.Module):

    def __init__(self, num_classes=20):
        super(BaselineCNN, self).__init__()

        # ----- Convolution Blocks -----
        self.conv_layers = nn.Sequential(

            # Conv Block 1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Conv Block 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Conv Block 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Compute flattened size dynamically by doing a dummy forward pass.
        # This means the Linear layer always matches your actual input shape
        # regardless of N_MELS, CLIP_DURATION, HOP_LENGTH, or sample rate.
        self._flat_size = self._get_flat_size()

        # ----- Fully Connected Layers -----
        self.fc_layers = nn.Sequential(
            nn.Linear(self._flat_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def _get_flat_size(self) -> int:
        """
        Run a single dummy tensor through conv_layers to find the exact
        flattened dimension.  Uses the same (N_MELS, T) shape that
        extract_logmel() produces based on config values.
        """
        import config
        T = int(config.CLIP_DURATION * config.SAMPLE_RATE / config.HOP_LENGTH)
        dummy = torch.zeros(1, 1, config.N_MELS, T)
        with torch.no_grad():
            out = self.conv_layers(dummy)
        return int(out.numel())         # numel() on a single-sample output

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)      # flatten keeping batch dimension
        x = self.fc_layers(x)
        return x