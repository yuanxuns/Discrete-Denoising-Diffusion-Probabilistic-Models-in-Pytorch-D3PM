"""Reference MNIST classifier used for conditional-generation evaluation."""

from torch import nn


class MNISTClassifier(nn.Module):
    """Small convolutional classifier for quantized MNIST images.

    Input:
        Float tensor of shape ``(B, 1, 28, 28)`` with values in ``[0, 1]``.

    Output:
        Logits tensor of shape ``(B, 10)`` for digit classes 0 through 9.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        """Classify a batch of MNIST images.

        Args:
            x: Float image tensor of shape ``(B, 1, 28, 28)``.

        Returns:
            Tensor of shape ``(B, 10)`` containing digit logits.
        """
        return self.classifier(self.features(x))
