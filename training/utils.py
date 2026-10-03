import os
import random
import logging
import math
import copy
from typing import Optional, Dict, Any, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer

logger = logging.getLogger(__name__)

def set_seed(seed: int) -> None:
    """
    Seeds Python, NumPy, PyTorch (CPU+CUDA), and sets torch deterministic flags.
    
    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EMA:
    """
    Exponential Moving Average of model parameters.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        """
        Creates a shadow copy of the model parameters.
        
        Args:
            model (nn.Module): The model to track.
            decay (float): The decay rate for the moving average.
        """
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.model = model
        
        # Register shadow copies
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        """
        Updates the shadow parameters.
        
        Args:
            model (nn.Module): The model with updated parameters.
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    assert name in self.shadow
                    new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                    self.shadow[name] = new_average.clone()

    def apply(self, model: nn.Module) -> None:
        """
        Copies shadow parameters into the model, backing up current parameters.
        
        Args:
            model (nn.Module): The model to update with EMA weights.
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    assert name in self.shadow
                    self.backup[name] = param.data.clone()
                    param.data = self.shadow[name]

    def restore(self, model: nn.Module) -> None:
        """
        Restores the original parameters from backup.
        
        Args:
            model (nn.Module): The model to restore.
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    assert name in self.backup
                    param.data = self.backup[name]
        self.backup = {}


class CosineWarmupScheduler:
    """
    Learning rate scheduler with linear warmup followed by cosine decay.
    """
    def __init__(self, optimizer: Optimizer, warmup_epochs: int, total_epochs: int, min_lr: float, steps_per_epoch: int):
        """
        Args:
            optimizer (Optimizer): The optimizer to schedule.
            warmup_epochs (int): Number of epochs for linear warmup.
            total_epochs (int): Total number of epochs for training.
            min_lr (float): Minimum learning rate after cosine decay.
            steps_per_epoch (int): Number of batches per epoch.
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        self.min_lr = min_lr
        self.current_step = 0
        
        # Store initial learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self) -> None:
        """
        Updates the optimizer learning rate. Called per-batch.
        """
        self.current_step += 1
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self._compute_lr(self.base_lrs[i])

    def _compute_lr(self, base_lr: float) -> float:
        """
        Computes the learning rate for the current step.
        """
        if self.current_step < self.warmup_steps:
            # Linear warmup
            alpha = self.current_step / max(1, self.warmup_steps)
            return self.min_lr + (base_lr - self.min_lr) * alpha
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, max(0.0, progress))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr + (base_lr - self.min_lr) * cosine_decay

    def get_lr(self) -> float:
        """
        Returns the current learning rate of the first parameter group.
        
        Returns:
            float: Current learning rate.
        """
        return self.optimizer.param_groups[0]['lr']


class EarlyStopping:
    """
    Monitors a metric and stops training when it stops improving.
    """
    def __init__(self, patience: int, mode: str = 'max', min_delta: float = 1e-4):
        """
        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            mode (str): One of 'min' or 'max'. In 'min' mode, training will stop when the quantity 
                        monitored has stopped decreasing.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, metric: float, model: nn.Module, path: str) -> bool:
        """
        Evaluates the metric and determines if training should stop.
        Saves the best model.
        
        Args:
            metric (float): The current metric value.
            model (nn.Module): The model to save if it improved.
            path (str): The file path to save the model.
            
        Returns:
            bool: True if training should stop, False otherwise.
        """
        score = metric if self.mode == 'max' else -metric
        
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model, path)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(model, path)
            self.counter = 0
            
        return self.early_stop

    def _save_checkpoint(self, model: nn.Module, path: str) -> None:
        """Saves model when metric improves."""
        torch.save(model.state_dict(), path)


class MetricTracker:
    """
    Accumulates metrics and computes epoch-level averages.
    """
    def __init__(self):
        self.reset()

    def update(self, val: float, batch_size: int = 1) -> None:
        """
        Updates the tracker with a new value.
        
        Args:
            val (float): The metric value (e.g., loss).
            batch_size (int): The batch size to weight the metric.
        """
        self.sum += val * batch_size
        self.count += batch_size

    def compute(self) -> float:
        """
        Computes the average value.
        
        Returns:
            float: The average metric.
        """
        if self.count == 0:
            return 0.0
        return self.sum / self.count

    def reset(self) -> None:
        """Resets the internal state."""
        self.sum = 0.0
        self.count = 0

