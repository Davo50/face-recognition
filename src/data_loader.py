import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

class FaceRecognitionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Custom dataset for face recognition
        
        Args:
            root_dir (str): Directory with all the images
            transform (callable, optional): Optional transform to be applied
        """
        self.image_dataset = datasets.ImageFolder(root_dir, transform=transform)
        self.transform = transform
        
    def __len__(self):
        return len(self.image_dataset)
    
    def __getitem__(self, idx):
        return self.image_dataset[idx]
    
    def get_class_to_idx(self):
        return self.image_dataset.class_to_idx

def create_data_loaders(data_dir, batch_size=32, split_ratio=0.2):
    """
    Create train and validation data loaders
    
    Args:
        data_dir (str): Path to dataset
        batch_size (int): Batch size for dataloaders
        split_ratio (float): Validation split ratio
    
    Returns:
        Tuple of (train_loader, val_loader, num_classes)
    """
    # Data augmentation and normalization
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Create datasets
    full_dataset = FaceRecognitionDataset(
        root_dir=data_dir, 
        transform=data_transforms['train']
    )
    
    # Get number of classes
    num_classes = len(full_dataset.get_class_to_idx())
    
    # Calculate dataset sizes
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * split_ratio)
    train_size = dataset_size - val_size
    
    # Split dataset
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size]
    )
    
    # Set different transforms for train and val
    train_dataset.dataset = full_dataset
    train_dataset.transform = data_transforms['train']
    val_dataset.dataset = full_dataset
    val_dataset.transform = data_transforms['val']
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader, num_classes, full_dataset.get_class_to_idx()

