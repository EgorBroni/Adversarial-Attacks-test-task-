from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class NormalizedClassifier(nn.Module):
    """Apply ImageNet normalization inside a classifier that accepts pixels in [0, 1]."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone((x - self.mean) / self.std)


def make_resnet18(num_classes: int, pretrained: bool = False) -> NormalizedClassifier:
    try:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
    except AttributeError:
        backbone = models.resnet18(pretrained=pretrained)
    backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
    return NormalizedClassifier(backbone)


def make_convnext_tiny(num_classes: int, pretrained: bool = False) -> NormalizedClassifier:
    try:
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = models.convnext_tiny(weights=weights)
    except AttributeError:
        backbone = models.convnext_tiny(pretrained=pretrained)
    in_features = backbone.classifier[-1].in_features
    backbone.classifier[-1] = nn.Linear(in_features, num_classes)
    return NormalizedClassifier(backbone)


def make_model(arch: str, num_classes: int, pretrained: bool) -> NormalizedClassifier:
    if arch == "resnet18":
        return make_resnet18(num_classes=num_classes, pretrained=pretrained)
    if arch == "convnext_tiny":
        return make_convnext_tiny(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown architecture {arch!r}; expected 'resnet18' or 'convnext_tiny'")
