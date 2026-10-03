import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionAggregator(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=128, num_classes=1):
        super().__init__()
        # Upgraded to Gated Attention Mechanism (CLAM style)
        # This significantly improves accuracy over simple attention by learning more robust tile weights
        self.attention_v = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh()
        )
        self.attention_u = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(hidden_dim, 1)
        
        # Added dropout for better regularization on the bag level
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(in_dim, num_classes)
        )
        
    def forward(self, h):
        """
        h: (batch_size, num_tiles, in_dim) or (num_tiles, in_dim)
        Returns slide-level logits and attention weights.
        """
        if h.dim() == 2:
            h = h.unsqueeze(0)
            
        # Gated mechanism: Tanh branch element-wise multiplied by Sigmoid branch
        A_v = self.attention_v(h)
        A_u = self.attention_u(h)
        A = self.attention_weights(A_v * A_u)  # (batch_size, num_tiles, 1)
        
        A = torch.transpose(A, 1, 2)  # (batch_size, 1, num_tiles)
        A = F.softmax(A, dim=2)  # Softmax over num_tiles
        
        M = torch.bmm(A, h)  # (batch_size, 1, in_dim)
        M = M.squeeze(1)     # (batch_size, in_dim)
        
        logits = self.classifier(M)
        return logits, A
