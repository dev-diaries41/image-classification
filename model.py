# model.py
import torch.nn as nn
import torchvision.models as models

class ImageClassifier(nn.Module):
    def __init__(self, num_classes, weights=models.ResNet18_Weights.DEFAULT):
        """
        num_classes: Number of output classes.
        weights: Weights of pre-trained model.
        """
        super(ImageClassifier, self).__init__()
        # Load a pre-trained ResNet18 model
        self.model = models.resnet18(weights=weights)
        # Replace the final fully-connected layer to match our number of classes.
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.model(x)


class ImageClassifierMobile(nn.Module):
    def __init__(self, num_classes, weights=models.MobileNet_V3_Large_Weights.DEFAULT):
        """
        num_classes: Number of output classes.
        weights: Weights of pre-trained model.
        """
        super(ImageClassifierMobile, self).__init__()
        # Load a pre-trained MobileNetV3 model
        self.model = models.mobilenet_v3_large(weights=weights)
        # MobileNetV3's classifier is a Sequential module.
        # Typically, it has a dropout and then a Linear layer. 
        # Replace the last Linear layer with a new one matching our num_classes.
        num_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(num_features, num_classes)

    
    def forward(self, x):
        return self.model(x)

