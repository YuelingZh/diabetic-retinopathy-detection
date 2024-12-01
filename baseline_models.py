import torch
import torch.nn as nn
from torchvision import models
# import timm


def get_vgg(num_classes=2, pretrained=True):
    """
    Initialize the VGG16 model.

    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to use pretrained weights.

    Returns:
        torch.nn.Module: VGG16 model.
    """
    model = models.vgg16(pretrained=pretrained)
    # Replace the classifier's last layer
    # model.classifier[6] = nn.Linear(4096, num_classes)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 1)  # Output one logit
    return model


def get_resnet(num_classes=2, pretrained=True):
    """
    Initialize the ResNet18 model.

    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to use pretrained weights.

    Returns:
        torch.nn.Module: ResNet18 model.
    """
    model = models.resnet18(pretrained=pretrained)
    # Replace the fully connected layer
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def get_densenet(num_classes=2, pretrained=True):
    """
    Initialize the DenseNet121 model.

    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to use pretrained weights.

    Returns:
        torch.nn.Module: DenseNet121 model.
    """
    model = models.densenet121(pretrained=pretrained)
    # Replace the classifier's last layer
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model


def get_efficientnet(num_classes=2, model_name="efficientnet_b0", pretrained=True):
    """
    Initialize an EfficientNet model.

    Args:
        num_classes (int): Number of output classes.
        model_name (str): EfficientNet model variant (e.g., "efficientnet_b0").
        pretrained (bool): Whether to use pretrained weights.

    Returns:
        torch.nn.Module: EfficientNet model.
    """
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model


def get_vit(num_classes=2, model_name="vit_base_patch16_224", pretrained=True):
    """
    Initialize a Vision Transformer (ViT) model.

    Args:
        num_classes (int): Number of output classes.
        model_name (str): ViT model variant (e.g., "vit_base_patch16_224").
        pretrained (bool): Whether to use pretrained weights.

    Returns:
        torch.nn.Module: ViT model.
    """
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model