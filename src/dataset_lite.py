"""
Fast Dataset Loader for KAIST Subset

Optimized for:
- Quick loading (10% of dataset)
- No preprocessing/alignment
- Simple augmentation
- Colab-friendly
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import numpy as np
from pathlib import Path


class FastKAISTDataset(Dataset):
    """
    Fast KAIST dataset loader using only set00 (10% of data)
    
    Features:
    - No alignment needed
    - Simple resize to 320x240
    - Basic augmentation
    - Fast loading
    """
    
    def __init__(
        self,
        root_dir: str,
        subset: str = 'set00',  # Use only set00 for speed
        split: str = 'train',  # train/val/test
        image_size: tuple = (320, 240),
        augment: bool = True
    ):
        self.root_dir = Path(root_dir)
        self.subset = subset
        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == 'train')
        
        # Find all RGB-Thermal pairs
        self.samples = self._load_samples()
        
        # Split data (70% train, 15% val, 15% test)
        self._split_data()
        
        # Transforms
        self.rgb_transform = self._get_rgb_transform()
        self.thermal_transform = self._get_thermal_transform()
        
    def _load_samples(self):
        """Load all RGB-Thermal image pairs from subset"""
        samples = []
        subset_path = self.root_dir / self.subset
        
        if not subset_path.exists():
            raise ValueError(f"Subset path not found: {subset_path}")
        
        # Iterate through video folders
        for video_dir in sorted(subset_path.glob('V*')):
            rgb_dir = video_dir / 'visible'
            thermal_dir = video_dir / 'lwir'
            
            if not (rgb_dir.exists() and thermal_dir.exists()):
                continue
            
            # Get all RGB images
            for rgb_path in sorted(rgb_dir.glob('*.jpg')):
                thermal_path = thermal_dir / rgb_path.name
                
                if thermal_path.exists():
                    samples.append({
                        'rgb': str(rgb_path),
                        'thermal': str(thermal_path),
                        'video': video_dir.name
                    })
        
        return samples
    
    def _split_data(self):
        """Split into train/val/test"""
        n = len(self.samples)
        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)
        
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)
        
        if self.split == 'train':
            self.samples = [self.samples[i] for i in indices[:train_end]]
        elif self.split == 'val':
            self.samples = [self.samples[i] for i in indices[train_end:val_end]]
        else:  # test
            self.samples = [self.samples[i] for i in indices[val_end:]]
    
    def _get_rgb_transform(self):
        """RGB image transforms"""
        transforms = [
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        if self.augment:
            transforms.insert(0, T.RandomHorizontalFlip(0.5))
            transforms.insert(1, T.ColorJitter(brightness=0.2, contrast=0.2))
        
        return T.Compose(transforms)
    
    def _get_thermal_transform(self):
        """Thermal image transforms"""
        transforms = [
            T.Resize(self.image_size),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ]
        
        if self.augment:
            transforms.insert(0, T.RandomHorizontalFlip(0.5))
        
        return T.Compose(transforms)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load images
        rgb = Image.open(sample['rgb']).convert('RGB')
        thermal = Image.open(sample['thermal'])
        
        # Apply transforms
        rgb = self.rgb_transform(rgb)
        thermal = self.thermal_transform(thermal)
        
        # Dummy labels (for now - simplified)
        labels = torch.tensor([1], dtype=torch.long)  # person class
        boxes = torch.zeros((4,), dtype=torch.float32)  # dummy bbox
        
        return {
            'rgb': rgb,
            'thermal': thermal,
            'labels': labels,
            'boxes': boxes
        }


def create_dataloaders(
    root_dir: str,
    batch_size: int = 16,
    num_workers: int = 2
):
    """Create train/val/test dataloaders"""
    
    train_dataset = FastKAISTDataset(root_dir, split='train', augment=True)
    val_dataset = FastKAISTDataset(root_dir, split='val', augment=False)
    test_dataset = FastKAISTDataset(root_dir, split='test', augment=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Dataset loaded:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset
    dataset = FastKAISTDataset('/kaggle/input/kaist-dataset', split='train')
    print(f"Dataset size: {len(dataset)}")
    
    # Test sample
    sample = dataset[0]
    print(f"RGB shape: {sample['rgb'].shape}")
    print(f"Thermal shape: {sample['thermal'].shape}")
    print("✓ Dataset test passed!")
