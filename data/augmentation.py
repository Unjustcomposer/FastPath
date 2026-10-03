"""
Data augmentation pipeline for histopathology images using torchvision v2.
"""
import torch
import torchvision.transforms.v2 as v2
from typing import Dict, Any


def get_normalization() -> v2.Transform:
    """
    Returns ImageNet mean and std normalisation transform.
    
    Returns:
        v2.Transform: Normalisation transform.
    """
    return v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def get_train_transforms(config: Dict[str, Any]) -> v2.Transform:
    """
    Get training transformations based on the given configuration.
    
    Args:
        config: Dictionary containing augmentation parameters (e.g., from train.yaml).
        
    Returns:
        v2.Transform: Composed training transformations.
    """
    transforms_list = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ]
    
    if config.get("random_flip", True):
        transforms_list.extend([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5)
        ])
        
    if config.get("random_rotation"):
        transforms_list.append(
            v2.RandomChoice([
                v2.RandomRotation((0, 0)),
                v2.RandomRotation((90, 90)),
                v2.RandomRotation((180, 180)),
                v2.RandomRotation((270, 270))
            ])
        )
        
    color_jitter = config.get("color_jitter", {})
    if color_jitter:
        transforms_list.append(
            v2.ColorJitter(
                brightness=color_jitter.get("brightness", 0.0),
                contrast=color_jitter.get("contrast", 0.0),
                saturation=color_jitter.get("saturation", 0.0),
                hue=color_jitter.get("hue", 0.0)
            )
        )
        
    blur_prob = config.get("gaussian_blur_prob", 0.0)
    if blur_prob > 0:
        transforms_list.append(
            v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=blur_prob)
        )
        
    transforms_list.append(get_normalization())
    
    cutout_prob = config.get("cutout_prob", 0.0)
    if cutout_prob > 0:
        cutout_size = config.get("cutout_size", 64)
        scale_min = (cutout_size * cutout_size) / (256 * 256) # Defaulting assumption
        transforms_list.append(
            v2.RandomErasing(p=cutout_prob, scale=(scale_min, scale_min*2), ratio=(1.0, 1.0))
        )
        
    return v2.Compose(transforms_list)

def get_eval_transforms() -> v2.Transform:
    """
    Get evaluation transformations (no randomness).
    
    Returns:
        v2.Transform: Composed evaluation transformations.
    """
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        get_normalization()
    ])
