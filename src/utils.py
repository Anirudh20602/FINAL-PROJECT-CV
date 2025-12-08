"""
Utility functions for fast training
"""

import torch
import time
from tqdm import tqdm


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device='cuda'):
    """Train for one epoch with mixed precision"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        # Move to device
        images = {
            'rgb': batch['rgb'].to(device),
            'thermal': batch['thermal'].to(device)
        }
        targets = {
            'labels': batch['labels'].to(device),
            'boxes': batch['boxes'].to(device)
        }
        
        # Mixed precision forward
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            loss = loss_dict['loss']
        
        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)


@torch.no_grad()
def validate(model, val_loader, criterion, device='cuda'):
    """Validate model"""
    model.eval()
    total_loss = 0
    
    for batch in tqdm(val_loader, desc='Validation'):
        images = {
            'rgb': batch['rgb'].to(device),
            'thermal': batch['thermal'].to(device)
        }
        targets = {
            'labels': batch['labels'].to(device),
            'boxes': batch['boxes'].to(device)
        }
        
        outputs = model(images)
        loss_dict = criterion(outputs, targets)
        total_loss += loss_dict['loss'].item()
    
    return total_loss / len(val_loader)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"✓ Checkpoint saved: {path}")


def load_checkpoint(model, optimizer, path, device='cuda'):
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']
