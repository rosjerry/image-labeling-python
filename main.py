#!/usr/bin/env python3
"""
Main training script for multi-label car classification.
Trains a model on car images with make, model, and steering wheel position labels.
"""

import argparse
import os
import json
import torch
import random
import numpy as np
from typing import Dict, Any

from data_loader import create_data_loaders
from model import create_model, count_parameters
from trainer import ModelTrainer


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train multi-label car classifier')
    
    # Data arguments
    parser.add_argument('--json_path', type=str, required=True,
                       help='Path to Label Studio JSON file')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory containing images')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio (default: 0.2)')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Image size for training (default: 224)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of worker processes (default: 4)')
    parser.add_argument('--pin_memory', action='store_true', default=True,
                       help='Pin memory for faster GPU transfer (default: True)')
    parser.add_argument('--persistent_workers', action='store_true',
                       help='Keep workers alive between epochs (default: False)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Number of steps to accumulate gradients (default: 1)')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Use mixed precision training (default: False)')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='resnet50',
                       choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                               'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                               'densenet121', 'densenet161', 'densenet169', 'densenet201'],
                       help='Backbone architecture (default: resnet50)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights (default: True)')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                       help='Dropout rate (default: 0.5)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=2,
                       help='Number of epochs (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    parser.add_argument('--scheduler_step', type=int, default=10,
                       help='Learning rate scheduler step size (default: 10)')
    parser.add_argument('--scheduler_gamma', type=float, default=0.1,
                       help='Learning rate scheduler gamma (default: 0.1)')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Early stopping patience (default: 10)')
    
    # Output arguments
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints (default: checkpoints)')
    parser.add_argument('--save_best', action='store_true', default=True,
                       help='Save best model (default: True)')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Logging interval (default: 10)')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use (default: auto)')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze backbone parameters (default: False)')
    
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    """Get the appropriate device."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    else:
        return torch.device(device_arg)


def save_config(args: argparse.Namespace, save_dir: str):
    """Save training configuration."""
    config_path = os.path.join(save_dir, 'config.json')
    config = vars(args)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def print_model_info(model: torch.nn.Module, num_classes: Dict[str, int]):
    """Print model information."""
    param_counts = count_parameters(model)
    
    print("\n" + "="*60)
    print("MODEL INFORMATION")
    print("="*60)
    print(f"Number of classes:")
    for attr_name, num_class in num_classes.items():
        print(f"  {attr_name}: {num_class}")
    
    print(f"\nModel parameters:")
    print(f"  Total parameters: {param_counts['total_params']:,}")
    print(f"  Trainable parameters: {param_counts['trainable_params']:,}")
    print(f"  Frozen parameters: {param_counts['frozen_params']:,}")
    print("="*60)


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save configuration
    save_config(args, args.save_dir)
    
    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader, num_classes = create_data_loaders(
        json_path=args.json_path,
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers
    )
    
    print(f"Data loaded successfully!")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Number of classes: {num_classes}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        backbone=args.backbone,
        num_classes=num_classes,
        pretrained=args.pretrained,
        dropout_rate=args.dropout_rate
    )
    
    # Freeze backbone if requested
    if args.freeze_backbone:
        model.freeze_backbone()
        print("Backbone frozen!")
    
    # Print model information
    print_model_info(model, num_classes)
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        device=device,
        save_dir=args.save_dir,
        log_interval=args.log_interval
    )
    
    # Train model
    print("\nStarting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_step=args.scheduler_step,
        scheduler_gamma=args.scheduler_gamma,
        early_stopping_patience=args.early_stopping_patience,
        save_best=args.save_best,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision
    )
    
    # Save training history
    trainer.save_history()
    
    # Plot training curves
    try:
        trainer.plot_training_curves()
    except Exception as e:
        print(f"Could not generate plots: {e}")
    
    print("\nTraining completed!")
    print(f"Best model saved to: {os.path.join(args.save_dir, 'best_model.pth')}")
    print(f"Training history saved to: {os.path.join(args.save_dir, 'training_history.json')}")


if __name__ == "__main__":
    main()
