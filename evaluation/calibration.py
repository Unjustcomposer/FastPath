import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TemperatureScaling(nn.Module):
    """
    Temperature Scaling for confidence calibration.
    Learns a single temperature parameter to scale logits before softmax.
    """
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        # Initialize temperature to 1.5
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Divide logits by temperature.
        
        Args:
            logits: Output logits from the model
            
        Returns:
            Scaled logits
        """
        return logits / self.temperature

def calibrate(model: nn.Module, val_loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    """
    Optimizes temperature parameter using NLL loss on validation data using LBFGS.
    
    Args:
        model: Trained model to calibrate
        val_loader: DataLoader for validation data
        device: Torch device (cpu/cuda)
        
    Returns:
        float: The optimal temperature value
    """
    model.eval()
    nll_criterion = nn.CrossEntropyLoss()
    
    logits_list = []
    labels_list = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            logits_list.append(logits.cpu())
            labels_list.append(labels.cpu())
            
    if not logits_list:
        logger.warning("Empty validation loader provided for calibration.")
        return 1.0
        
    all_logits = torch.cat(logits_list).to(device)
    all_labels = torch.cat(labels_list).to(device)
    
    temp_model = TemperatureScaling().to(device)
    optimizer = optim.LBFGS([temp_model.temperature], lr=0.01, max_iter=50)
    
    def eval_loss():
        optimizer.zero_grad()
        loss = nll_criterion(temp_model(all_logits), all_labels)
        loss.backward()
        return loss
        
    optimizer.step(eval_loss)
    
    optimal_temp = temp_model.temperature.item()
    logger.info(f"Optimal temperature found: {optimal_temp:.4f}")
    return optimal_temp

def apply_calibration(logits: np.ndarray, temperature: float) -> np.ndarray:
    """
    Applies temperature scaling to numpy logits and returns calibrated probabilities.
    
    Args:
        logits: Uncalibrated logits
        temperature: Learned temperature value
        
    Returns:
        np.ndarray: Calibrated probabilities
    """
    if temperature <= 0:
        logger.warning("Invalid temperature (<= 0). Defaulting to 1.0")
        temperature = 1.0
        
    calibrated_logits = logits / temperature
    
    # Numerically stable softmax
    exp_logits = np.exp(calibrated_logits - np.max(calibrated_logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    return probs
