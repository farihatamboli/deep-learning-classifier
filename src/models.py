import torch.nn as nn
import torchvision.models as models


class LogisticRegressionBaseline(nn.Module):
    def __init__(self, in_features: int, dropout_p: float = 0.3):
        super().__init__()
        self.fc = nn.Linear(in_features, 2)
        self.drop = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.drop(x)
        return self.fc(x)


class SimpleCNN(nn.Module):
    def __init__(self, dropout_p: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(64 * 64 * 64, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        return self.classifier(x)


def build_resnet18(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m
