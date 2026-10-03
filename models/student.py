import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class StudentModel(nn.Module):
    def __init__(self, num_classes=1, pretrained=True, feature_dim=128, teacher_dim=2048):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Main compact projection (this is the 128-d output used for fast inference)
        self.projection = nn.Sequential(
            nn.Linear(in_features, feature_dim),
            nn.GELU()
        )
        
        # Feature-hint projection (only used during KD training to map the 128-d vector 
        # back up to the teacher's heavy dimension (e.g., 2048) for the MSE hint loss)
        self.hint_projection = nn.Sequential(
            nn.Linear(feature_dim, teacher_dim)
        )
        
        self.classifier = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        proj_features = self.projection(features)
        logits = self.classifier(proj_features)
        
        # Only return hints during training to avoid breaking inference/ONNX export
        if self.training:
            hint_features = self.hint_projection(proj_features)
            return logits, proj_features, hint_features
            
        return logits, proj_features
