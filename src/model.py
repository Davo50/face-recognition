#model.py
import torch
import torch.nn as nn
import torchvision.models as models

class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        """
        Face recognition model using transfer learning
        
        Args:
            num_classes (int): Number of unique people to recognize
            pretrained (bool): Use pretrained weights
        """
        super(FaceRecognitionModel, self).__init__()
        
        # Load pre-trained ResNet50
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        
        # Freeze base model parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Replace the last fully connected layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input image tensor
        
        Returns:
            torch.Tensor: Class predictions
        """
        return self.backbone(x)
    
    def unfreeze_backbone(self, layers_to_unfreeze=None):
        """
        Unfreeze specified layers for fine-tuning
        
        Args:
            layers_to_unfreeze (int, optional): Number of layers to unfreeze from the end
        """
        if layers_to_unfreeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # Unfreeze specified layers
            for param in list(self.backbone.parameters())[-layers_to_unfreeze:]:
                param.requires_grad = True

                