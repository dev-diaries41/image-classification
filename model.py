import torch.nn as nn
import torchvision.models as models
import torch
from hebbian_mlp import HebbianMLP

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
    

class ImageClassifierWithMLP(nn.Module):
    def __init__(self, num_classes, backbone='resnet', mlp_hidden=256):
        super().__init__()
        if backbone == 'resnet':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # remove final fc
        elif backbone == 'mobilenet':
            self.backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Identity()
        else:
            raise ValueError("Backbone not supported")
        
        gate_threshold = torch.sigmoid(torch.tensor(1)).item()
        self.mlp = HebbianMLP(layer_sizes=[in_features, mlp_hidden, num_classes], gate_fn=self.gate_fn, gate_threshold = gate_threshold)
        
    def forward(self, x, return_activations=False):
        features = self.backbone(x)
        return self.mlp(features, return_activations=return_activations)
    
    
    def gate_fn(self, module, x, **kwargs):
        loss = kwargs.get('loss', None)
        avg_loss = kwargs.get('avg_loss', None)
        if loss is not None and avg_loss is not None:
            gate = torch.sigmoid(torch.tensor(avg_loss / loss, dtype=torch.float32, device=x.device))
            return gate
        print("Warning: loss and avg loss not passed")
        return 1.0