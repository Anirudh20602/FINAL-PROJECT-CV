"""
Fast Pedestrian Detection - Lightweight Version

Optimized for Google Colab training with limited data
"""

__version__ = '1.0.0-lite'

from .model_lite import FastPedestrianDetector, SimpleLoss, create_model
from .dataset_lite import FastKAISTDataset, create_dataloaders
# from .utils import train_one_epoch, validate, save_checkpoint, load_checkpoint

__all__ = [
    'FastPedestrianDetector',
    'SimpleLoss',
    'create_model',
    'FastKAISTDataset',
    'create_dataloaders',
    'train_one_epoch',
    'validate',
    'save_checkpoint',
    'load_checkpoint',
]
