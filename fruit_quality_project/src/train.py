"""
Training module for fruit quality classification.
Implements transfer learning with MobileNetV2/ResNet.
"""

import os
from typing import Tuple, Dict, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from .utils import get_device, print_section, count_parameters


def create_model(
    model_name: str = "mobilenetv2",
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True
) -> nn.Module:
    """
    Create a classification model using transfer learning.
    
    Args:
        model_name: 'mobilenetv2', 'resnet18', or 'resnet50'
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        freeze_backbone: Whether to freeze backbone layers initially
        
    Returns:
        PyTorch model
    """
    weights = "IMAGENET1K_V1" if pretrained else None
    
    if model_name == "mobilenetv2":
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        backbone = model.features
        
    elif model_name == "resnet18":
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        backbone = nn.Sequential(*list(model.children())[:-1])
        
    elif model_name == "resnet50":
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        backbone = nn.Sequential(*list(model.children())[:-1])
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Freeze backbone if requested
    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
        print(f"[INFO] Backbone frozen. Training only classifier head.")
    
    trainable_params = count_parameters(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model: {model_name}")
    print(f"[INFO] Total parameters: {total_params:,}")
    print(f"[INFO] Trainable parameters: {trainable_params:,}")
    
    return model


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = running_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    """
    Validate for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]")
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    avg_loss = running_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 20,
    learning_rate: float = 0.001,
    save_path: Optional[str] = None,
    unfreeze_after: int = 5,
    device: Optional[torch.device] = None
) -> Dict:
    """
    Complete training loop with optional backbone unfreezing.
    
    Args:
        model: PyTorch model
        train_loader: Training dataloader
        test_loader: Test/validation dataloader
        epochs: Number of training epochs
        learning_rate: Initial learning rate
        save_path: Path to save best model
        unfreeze_after: Epoch after which to unfreeze backbone
        device: Device to train on
        
    Returns:
        Dictionary with training history
    """
    if device is None:
        device = get_device()
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=3)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'best_val_acc': 0.0
    }
    
    print_section("Starting Training")
    
    for epoch in range(epochs):
        # Unfreeze backbone after certain epochs for fine-tuning
        if epoch == unfreeze_after:
            print(f"\n[INFO] Unfreezing backbone for fine-tuning...")
            for param in model.parameters():
                param.requires_grad = True
            
            # Lower learning rate for fine-tuning
            optimizer = optim.Adam(model.parameters(), lr=learning_rate * 0.1)
        
        # Training
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validation
        val_loss, val_acc = validate_epoch(
            model, test_loader, criterion, device, epoch
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Log results
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > history['best_val_acc']:
            history['best_val_acc'] = val_acc
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, save_path)
                print(f"  [SAVED] Best model saved to {save_path}")
    
    print_section("Training Complete")
    print(f"Best Validation Accuracy: {history['best_val_acc']:.2f}%")
    
    return history


def load_model(model_path: str, model_name: str = "mobilenetv2", num_classes: int = 2) -> nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        model_name: Architecture name
        num_classes: Number of classes
        
    Returns:
        Loaded model
    """
    model = create_model(model_name, num_classes, pretrained=False, freeze_backbone=False)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"[INFO] Loaded model from {model_path}")
    print(f"[INFO] Checkpoint validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    
    return model
