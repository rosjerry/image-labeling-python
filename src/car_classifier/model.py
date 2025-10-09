import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, List, Optional
import timm


class MultiLabelCarClassifier(nn.Module):
    """
    Multi-label car classifier using pre-trained backbone.
    Supports ResNet, EfficientNet, and other architectures.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: Dict[str, int] = None,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
    ):
        """
        Initialize the multi-label classifier.

        Args:
            backbone: Backbone architecture ('resnet50', 'efficientnet_b0', etc.)
            num_classes: Dictionary with number of classes for each attribute
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for classification heads
        """
        super(MultiLabelCarClassifier, self).__init__()

        self.backbone_name = backbone
        self.num_classes = num_classes or {"make": 10, "model": 20, "steer_wheel": 2}

        # Create backbone
        self.backbone = self._create_backbone(backbone, pretrained)

        # Get feature dimension
        self.feature_dim = self._get_feature_dim()

        # Create classification heads
        self.classifiers = nn.ModuleDict(
            {
                "make": nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(self.feature_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(512, self.num_classes["make"]),
                ),
                "model": nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(self.feature_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(512, self.num_classes["model"]),
                ),
                "steer_wheel": nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(self.feature_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(256, self.num_classes["steer_wheel"]),
                ),
            }
        )

    def _create_backbone(self, backbone: str, pretrained: bool) -> nn.Module:
        """Create backbone network."""
        if backbone.startswith("resnet"):
            if backbone == "resnet18":
                model = models.resnet18(pretrained=pretrained)
            elif backbone == "resnet34":
                model = models.resnet34(pretrained=pretrained)
            elif backbone == "resnet50":
                model = models.resnet50(pretrained=pretrained)
            elif backbone == "resnet101":
                model = models.resnet101(pretrained=pretrained)
            elif backbone == "resnet152":
                model = models.resnet152(pretrained=pretrained)
            else:
                raise ValueError(f"Unsupported ResNet variant: {backbone}")

            # Remove the final classification layer
            return nn.Sequential(*list(model.children())[:-1])

        elif backbone.startswith("efficientnet"):
            # Use timm for EfficientNet
            model = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
            return model

        elif backbone.startswith("densenet"):
            if backbone == "densenet121":
                model = models.densenet121(pretrained=pretrained)
            elif backbone == "densenet161":
                model = models.densenet161(pretrained=pretrained)
            elif backbone == "densenet169":
                model = models.densenet169(pretrained=pretrained)
            elif backbone == "densenet201":
                model = models.densenet201(pretrained=pretrained)
            else:
                raise ValueError(f"Unsupported DenseNet variant: {backbone}")

            # Remove the final classification layer
            return nn.Sequential(*list(model.children())[:-1])

        else:
            # Try timm for other architectures
            try:
                model = timm.create_model(
                    backbone, pretrained=pretrained, num_classes=0
                )
                return model
            except:
                raise ValueError(f"Unsupported backbone: {backbone}")

    def _get_feature_dim(self) -> int:
        """Get feature dimension from backbone."""
        # Create a dummy input to get feature dimension
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = self.backbone(dummy_input)
            if isinstance(features, tuple):
                features = features[0]
            return features.view(features.size(0), -1).size(1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, height, width)

        Returns:
            Dictionary with predictions for each attribute
        """
        # Extract features
        features = self.backbone(x)
        if isinstance(features, tuple):
            features = features[0]

        # Flatten features
        features = features.view(features.size(0), -1)

        # Get predictions for each attribute
        predictions = {}
        for attr_name, classifier in self.classifiers.items():
            predictions[attr_name] = classifier(features)

        return predictions

    def get_feature_extractor(self) -> nn.Module:
        """Get feature extractor (backbone only)."""
        return self.backbone

    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True


class MultiLabelLoss(nn.Module):
    """
    Multi-label loss function combining cross-entropy losses for each attribute.
    """

    def __init__(self, loss_weights: Optional[Dict[str, float]] = None):
        """
        Initialize multi-label loss.

        Args:
            loss_weights: Optional weights for each attribute loss
        """
        super(MultiLabelLoss, self).__init__()
        self.loss_weights = loss_weights or {
            "make": 1.0,
            "model": 1.0,
            "steer_wheel": 1.0,
        }
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-label loss.

        Args:
            predictions: Model predictions for each attribute
            targets: Ground truth labels for each attribute

        Returns:
            Dictionary with individual losses and total loss
        """
        losses = {}
        total_loss = 0.0

        for attr_name in predictions.keys():
            if attr_name in targets:
                loss = self.cross_entropy(predictions[attr_name], targets[attr_name])
                weighted_loss = loss * self.loss_weights.get(attr_name, 1.0)
                losses[f"{attr_name}_loss"] = loss
                losses[f"{attr_name}_weighted_loss"] = weighted_loss
                total_loss += weighted_loss

        losses["total_loss"] = total_loss
        return losses


def create_model(
    backbone: str = "resnet50",
    num_classes: Dict[str, int] = None,
    pretrained: bool = True,
    dropout_rate: float = 0.5,
) -> MultiLabelCarClassifier:
    """
    Create a multi-label car classifier model.

    Args:
        backbone: Backbone architecture
        num_classes: Number of classes for each attribute
        pretrained: Whether to use pretrained weights
        dropout_rate: Dropout rate for classification heads

    Returns:
        MultiLabelCarClassifier model
    """
    return MultiLabelCarClassifier(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
    )


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count trainable and total parameters in the model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
    }


if __name__ == "__main__":
    # Test the model
    num_classes = {"make": 5, "model": 10, "steer_wheel": 2}

    # Test different backbones
    backbones = ["resnet50", "efficientnet_b0", "densenet121"]

    for backbone in backbones:
        try:
            print(f"\nTesting {backbone}:")
            model = create_model(backbone=backbone, num_classes=num_classes)

            # Test forward pass
            dummy_input = torch.randn(2, 3, 224, 224)
            predictions = model(dummy_input)

            print(f"  Input shape: {dummy_input.shape}")
            for attr_name, pred in predictions.items():
                print(f"  {attr_name} prediction shape: {pred.shape}")

            # Count parameters
            param_counts = count_parameters(model)
            print(f"  Total parameters: {param_counts['total_params']:,}")
            print(f"  Trainable parameters: {param_counts['trainable_params']:,}")

        except Exception as e:
            print(f"  Error with {backbone}: {e}")

    # Test loss function
    print("\nTesting loss function:")
    loss_fn = MultiLabelLoss()

    # Create dummy predictions and targets
    predictions = {
        "make": torch.randn(2, num_classes["make"]),
        "model": torch.randn(2, num_classes["model"]),
        "steer_wheel": torch.randn(2, num_classes["steer_wheel"]),
    }

    targets = {
        "make": torch.randint(0, num_classes["make"], (2,)),
        "model": torch.randint(0, num_classes["model"], (2,)),
        "steer_wheel": torch.randint(0, num_classes["steer_wheel"], (2,)),
    }

    losses = loss_fn(predictions, targets)
    print(f"Losses: {losses}")
