"""Reusable pretrained ResNet18 feature extractor."""
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.feature_size = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return torch.flatten(features, 1)
