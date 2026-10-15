import torch
import torch.nn as nn
import timm
import logging

class TeacherModel(nn.Module):
    def __init__(self, model_name='resnet50', weights_path=None, pretrained=True):
        super().__init__()
        self.model_name = model_name.lower()
        
        if self.model_name == 'resnet50':
            self.model = timm.create_model('resnet50', pretrained=pretrained, num_classes=0)
            self.feature_dim = 2048
            
        elif self.model_name == 'uni':
            try:
                self.model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=pretrained, init_values=1e-5, dynamic_img_size=True)
                if hasattr(self.model, 'reset_classifier'):
                    self.model.reset_classifier(0)
                self.feature_dim = 1024
            except Exception as e:
                logging.error("Failed to load UNI model. Run: huggingface-cli login")
                raise
                
        elif self.model_name == 'ctranspath':
            self.model = timm.create_model('swin_tiny_patch4_window7_224', img_size=224, embed_dim=96,
                                           depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7,
                                           num_classes=0)
            self.feature_dim = 768
            if weights_path:
                state_dict = torch.load(weights_path, map_location='cpu')
                if 'model' in state_dict: state_dict = state_dict['model']
                elif 'state_dict' in state_dict: state_dict = state_dict['state_dict']
                self.model.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        self.classifier = nn.Linear(self.feature_dim, 1)

    def forward(self, x):
        features = self.model(x)
        logits = self.classifier(features)
        return logits, features

