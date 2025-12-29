import torch.nn as nn
import torchvision.models as models

class ImageClassifier(nn.Module):
    def __init__(self, num_classes, architecture='resnet', weights=None):
        """
        num_classes: Number of output classes.
        architecture: 'resnet' or 'mobilenet'.
        weights: Pre-trained weights. If None, defaults will be used.
        """
        super(ImageClassifier, self).__init__()

        if architecture.lower() == 'resnet':
            weights = weights or models.ResNet18_Weights.DEFAULT
            self.model = models.resnet18(weights=weights)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)

        elif architecture.lower() == 'mobilenet':
            weights = weights or models.MobileNet_V3_Large_Weights.DEFAULT
            self.model = models.mobilenet_v3_large(weights=weights)
            num_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(num_features, num_classes)

        else:
            raise ValueError("architecture must be 'resnet' or 'mobilenet'")

    def forward(self, x):
        return self.model(x)
    