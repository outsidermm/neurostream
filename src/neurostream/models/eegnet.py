"""
EEGNet: Compact CNN for EEG-based BCIs.
Lawhern et al. (2018) — https://doi.org/10.1088/1741-2552/aace8c

Architecture (for 22-channel, 250 Hz, 4-class):
  Block 1: Temporal conv  →  Depthwise spatial conv  →  BN  →  ELU  →  AvgPool  →  Dropout
  Block 2: Depthwise-separable conv  →  BN  →  ELU  →  AvgPool  →  Dropout
  Classifier: Flatten  →  Linear

All hyperparameter defaults match Table I of the paper.
"""

import torch
import torch.nn as nn


def _renorm_max_norm(
    weight: torch.Tensor,
    max_value: float,
    dim: int | tuple[int, ...],
) -> None:
    """In-place Keras-style max_norm constraint per slice along `dim`.

    Rescales weight slices whose L2 norm exceeds max_value so the new norm equals
    max_value; slices already under the cap are untouched. Matches the renorm
    Keras applies after every optimiser step.
    """
    with torch.no_grad():
        norms = weight.norm(p=2, dim=dim, keepdim=True).clamp(min=1e-8)
        scale = (max_value / norms).clamp(max=1.0)
        weight.mul_(scale)


class EEGNet(nn.Module):
    def __init__(
        self,
        n_classes: int = 4,
        n_channels: int = 22,
        n_samples: int = 750,
        fs: int = 250,
        f1: int = 8,  # temporal filter size
        d: int = 2,  # depth multiplier
        dropout: float = 0.25,
        kernel_length: int | None = None,
    ):
        super().__init__()
        f2 = f1 * d  # number of pointwise filters
        # Lawhern's official BCI IV 2a config uses 32 samples at 128 Hz (250 ms).
        # Default falls back to fs/2 to preserve the legacy heuristic.
        temporal_kernal = kernel_length if kernel_length is not None else fs // 2

        self.temporal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=f1,
            kernel_size=(1, temporal_kernal),
            padding=(0, temporal_kernal // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(f1)
        self.depthwise_conv1 = nn.Conv2d(
            in_channels=f1,
            out_channels=f1 * d,
            kernel_size=(n_channels, 1),
            groups=f1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(f1 * d)
        self.elu1 = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(dropout)

        self.depthwise_conv2 = nn.Conv2d(
            in_channels=f1 * d,
            out_channels=f1 * d,
            kernel_size=(1, 16),
            padding=(0, 8),
            groups=f1 * d,
            bias=False,
        )

        self.pointwise_conv1 = nn.Conv2d(
            in_channels=f1 * d, out_channels=f2, kernel_size=(1, 1), bias=False
        )
        self.bn3 = nn.BatchNorm2d(f2)
        self.elu2 = nn.ELU()
        self.avg_pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            in_features=self._get_classifier_in_features(n_channels, n_samples),
            out_features=n_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, T) — raw EEG epochs, float32
        Returns:
            logits: (N, n_classes) — unnormalised scores
        """
        # Add channel dimension expected by Conv2d: (N, 1, C, T)
        x = x.unsqueeze(1)

        # Block 1
        x = self.temporal_conv(x)
        x = self.bn1(x)
        x = self.depthwise_conv1(x)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.avg_pool1(x)
        x = self.dropout1(x)

        # Block 2
        x = self.depthwise_conv2(x)
        x = self.pointwise_conv1(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.avg_pool2(x)
        x = self.dropout2(x)

        # Classifier
        x = x.flatten(start_dim=1)
        return self.classifier(x)

    def apply_max_norm(self) -> None:
        """Apply Keras-style max_norm constraints. Call after every optimizer.step().

        - Depthwise spatial conv: max norm 1.0 per output filter (weight shape
          (F1*D, 1, n_channels, 1) → renorm across the last three dims).
        - Final classifier: max norm 0.25 per output unit (Linear weight shape
          (n_classes, features) → renorm across input features).
        """
        _renorm_max_norm(self.depthwise_conv1.weight, max_value=1.0, dim=(1, 2, 3))
        _renorm_max_norm(self.classifier.weight, max_value=0.25, dim=1)

    def _get_classifier_in_features(self, n_channels: int, n_samples: int) -> int:
        """Dry-run a zero tensor through the feature extractor to get output size."""
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            dummy = self.temporal_conv(dummy)
            dummy = self.bn1(dummy)
            dummy = self.depthwise_conv1(dummy)
            dummy = self.bn2(dummy)
            dummy = self.avg_pool1(dummy)
            dummy = self.depthwise_conv2(dummy)
            dummy = self.pointwise_conv1(dummy)
            dummy = self.bn3(dummy)
            dummy = self.avg_pool2(dummy)
        return int(dummy.flatten(start_dim=1).shape[1])
