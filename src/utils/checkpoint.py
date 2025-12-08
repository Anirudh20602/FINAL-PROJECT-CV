"""
Checkpoint management utilities
"""

import os
import torch
from typing import Dict, Any, Optional
import logging


def save_checkpoint(
    state: Dict[str, Any],
    save_dir: str,
    filename: str = "checkpoint.pth",
    is_best: bool = False
) -> None:
    """
    Save model checkpoint
    
    Args:
        state: State dictionary containing model, optimizer, etc.
        save_dir: Directory to save checkpoint
        filename: Checkpoint filename
        is_best: Whether this is the best model so far
    """
    os.makedirs(save_dir, exist_ok=True)
    
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = os.path.join(save_dir, "best.pth")
        torch.save(state, best_filepath)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Load model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        scheduler: Optional scheduler to load state
        device: Device to load checkpoint to
        
    Returns:
        Dictionary with epoch, best_metric, etc.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'best_metric': checkpoint.get('best_metric', 0.0),
        'config': checkpoint.get('config', None)
    }


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Get path to latest checkpoint in directory
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to latest checkpoint or None
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [
        f for f in os.listdir(checkpoint_dir)
        if f.endswith('.pth') and f != 'best.pth'
    ]
    
    if not checkpoints:
        return None
    
    # Sort by modification time
    checkpoints.sort(
        key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)),
        reverse=True
    )
    
    return os.path.join(checkpoint_dir, checkpoints[0])


def cleanup_checkpoints(
    checkpoint_dir: str,
    keep_last: int = 5,
    keep_best: bool = True
) -> None:
    """
    Remove old checkpoints, keeping only the most recent ones
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last: Number of recent checkpoints to keep
        keep_best: Whether to keep best.pth
    """
    if not os.path.exists(checkpoint_dir):
        return
    
    checkpoints = [
        f for f in os.listdir(checkpoint_dir)
        if f.endswith('.pth') and (f != 'best.pth' or not keep_best)
    ]
    
    if len(checkpoints) <= keep_last:
        return
    
    # Sort by modification time
    checkpoints.sort(
        key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)),
        reverse=True
    )
    
    # Remove old checkpoints
    for checkpoint in checkpoints[keep_last:]:
        os.remove(os.path.join(checkpoint_dir, checkpoint))
