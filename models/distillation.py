import torch
import torch.nn as nn

class KDLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.1, temperature=4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta  # Weight for Feature-Hint Loss
        self.T = temperature
        self.bce = nn.BCEWithLogitsLoss()
        self.soft_bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        
    def forward(self, student_logits, teacher_logits, true_labels, student_hints=None, teacher_feats=None):
        if student_logits.dim() == 1:
            student_logits = student_logits.unsqueeze(-1)
        if true_labels.dim() == 1:
            true_labels = true_labels.unsqueeze(-1)
        true_labels = true_labels.float()

        # Hard loss (on ground truth or pseudo-labels)
        hard_loss = self.bce(student_logits, true_labels)
        
        # Soft loss: seamlessly handle both 1-class and 2-class teacher logits
        if teacher_logits.dim() == 1:
            teacher_logits = teacher_logits.unsqueeze(-1)

        if teacher_logits.shape[-1] == 2:
            # 2-class output: [negative, positive] -> take softmax probability of positive class
            soft_targets = torch.softmax(teacher_logits / self.T, dim=-1)[:, 1:]
        else:
            # 1-class binary logit
            soft_targets = torch.sigmoid(teacher_logits / self.T)

        soft_loss = self.soft_bce(student_logits / self.T, soft_targets)
        
        # Feature Hint Loss: MSE to force student embeddings to map exactly to teacher embeddings
        hint_loss = 0.0
        if student_hints is not None and teacher_feats is not None:
            hint_loss = self.mse(student_hints, teacher_feats)
            
        return self.alpha * hard_loss + (1 - self.alpha) * (soft_loss * (self.T ** 2)) + self.beta * hint_loss
