import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import copy
from src.model import FaceRecognitionModel
from src.data_loader import create_data_loaders

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    """
    Train the face recognition model
    
    Args:
        model (nn.Module): Neural network model
        dataloaders (dict): Dictionary of data loaders
        criterion (nn.Module): Loss function
        optimizer (torch.optim): Optimizer
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler
        num_epochs (int): Number of training epochs
    
    Returns:
        Trained model
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it has the best validation accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

def main():
    # Prepare data
    train_loader, val_loader, num_classes, class_to_idx = create_data_loaders('data/people_dataset')
    
    # Create model
    model = FaceRecognitionModel(num_classes)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Learning rate scheduler
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Combine dataloaders
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    # Train model
    trained_model = train_model(
        model, 
        dataloaders, 
        criterion, 
        optimizer, 
        exp_lr_scheduler, 
        num_epochs=25
    )
    
    # Save model
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'class_to_idx': class_to_idx,
        'num_classes': num_classes
    }, 'face_recognition_model.pth')

if __name__ == '__main__':
    main()


