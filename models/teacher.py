import torch
import torch.nn as nn
import timm
import logging

class TeacherModel(nn.Module):
    def __init__(self, model_name='resnet50', weights_path=None):
        super().__init__()
        self.model_name = model_name.lower()
        
        if self.model_name == 'resnet50':
            self.model = timm.create_model('resnet50', pretrained=True, num_classes=0)
            
        elif self.model_name == 'uni':
            try:
                # UNI requires hf login
                self.model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
                # Ensure no classifier head is used
                if hasattr(self.model, 'reset_classifier'):
                    self.model.reset_classifier(0)
            except Exception as e:
                logging.error("Failed to load UNI model from Hugging Face Hub. "
                              "Please ensure you have run `huggingface-cli login` and have been granted access to MahmoodLab/uni.")
                logging.error(f"Error details: {e}")
                raise
                
        elif self.model_name == 'ctranspath':
            self.model = timm.create_model('swin_tiny_patch4_window7_224', img_size=224, embed_dim=96, 
                                           depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, 
                                           num_classes=0)
            if weights_path:
                try:
                    state_dict = torch.load(weights_path, map_location='cpu')
                    # Handle cases where the state dict is nested under 'model' or 'state_dict'
                    if 'model' in state_dict:
                        state_dict = state_dict['model']
                    elif 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                    self.model.load_state_dict(state_dict, strict=False)
                    logging.info(f"Loaded CTransPath weights from {weights_path}")
                except Exception as e:
                    logging.error(f"Failed to load CTransPath weights from {weights_path}. Error: {e}")
            else:
                logging.warning("No weights path provided for CTransPath. Model initialized with random weights.")
                
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

    def forward(self, x):
        return self.model(x)
