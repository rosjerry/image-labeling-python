import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import time
from collections import defaultdict


class EarlyStopping:
    """Early stopping utility to stop training when validation loss stops improving."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights
            
        Returns:
            True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                self.restore_checkpoint(model)
            return True
        return False
    
    def save_checkpoint(self, model: nn.Module):
        """Save model checkpoint."""
        self.best_weights = model.state_dict().copy()
    
    def restore_checkpoint(self, model: nn.Module):
        """Restore best model weights."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class ModelTrainer:
    """
    Trainer class for multi-label car classification.
    Handles training, validation, and model checkpointing.
    """
    
    def __init__(self, model: nn.Module, device: torch.device, 
                 save_dir: str = 'checkpoints', log_interval: int = 10):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            device: Device to use for training
            save_dir: Directory to save checkpoints
            log_interval: Interval for logging
        """
        self.model = model
        self.device = device
        self.save_dir = save_dir
        self.log_interval = log_interval
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': defaultdict(list),
            'val_acc': defaultdict(list)
        }
        
        # Move model to device
        self.model.to(device)
    
    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, 
                   criterion: nn.Module, epoch: int, gradient_accumulation_steps: int = 1,
                   scaler=None) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            epoch: Current epoch number
            gradient_accumulation_steps: Number of steps to accumulate gradients
            scaler: Mixed precision scaler (optional)
            
        Returns:
            Tuple of (average_loss, accuracies)
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        # Track accuracies for each attribute
        correct = defaultdict(int)
        total = defaultdict(int)
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            labels = {k: v.to(self.device) for k, v in labels.items()}
            
            # Forward pass with mixed precision if enabled
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    predictions = self.model(images)
                    losses = criterion(predictions, labels)
                    loss = losses['total_loss'] / gradient_accumulation_steps
            else:
                predictions = self.model(images)
                losses = criterion(predictions, labels)
                loss = losses['total_loss'] / gradient_accumulation_steps
            
            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights only after accumulating gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item()
            
            # Compute accuracies
            with torch.no_grad():
                for attr_name in predictions.keys():
                    if attr_name in labels:
                        pred_labels = torch.argmax(predictions[attr_name], dim=1)
                        correct[attr_name] += (pred_labels == labels[attr_name]).sum().item()
                        total[attr_name] += labels[attr_name].size(0)
            
            # Update progress bar
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}'
                })
        
        # Calculate average metrics
        avg_loss = total_loss / num_batches
        accuracies = {attr_name: correct[attr_name] / total[attr_name] 
                     for attr_name in correct.keys()}
        
        return avg_loss, accuracies
    
    def validate_epoch(self, val_loader: DataLoader, criterion: nn.Module, 
        epoch: int) -> Tuple[float, Dict[str, float]]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            epoch: Current epoch number
            
        Returns:
            Tuple of (average_loss, accuracies)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        # Track accuracies for each attribute
        correct = defaultdict(int)
        total = defaultdict(int)
        
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(pbar):
                # Move to device
                images = images.to(self.device)
                labels = {k: v.to(self.device) for k, v in labels.items()}
                
                # Forward pass
                predictions = self.model(images)
                
                # Compute loss
                losses = criterion(predictions, labels)
                loss = losses['total_loss']
                
                # Update metrics
                total_loss += loss.item()
                
                # Compute accuracies
                for attr_name in predictions.keys():
                    if attr_name in labels:
                        pred_labels = torch.argmax(predictions[attr_name], dim=1)
                        correct[attr_name] += (pred_labels == labels[attr_name]).sum().item()
                        total[attr_name] += labels[attr_name].size(0)
                
                # Update progress bar
                if batch_idx % self.log_interval == 0:
                    pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}'
                    })
        
        # Calculate average metrics
        avg_loss = total_loss / num_batches
        accuracies = {attr_name: correct[attr_name] / total[attr_name] 
                     for attr_name in correct.keys()}
        
        return avg_loss, accuracies
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = 100, learning_rate: float = 1e-4,
              weight_decay: float = 1e-4, scheduler_step: int = 10,
              scheduler_gamma: float = 0.1, early_stopping_patience: int = 10,
              save_best: bool = True, gradient_accumulation_steps: int = 1,
              mixed_precision: bool = False) -> Dict:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            scheduler_step: Step size for learning rate scheduler
            scheduler_gamma: Gamma for learning rate scheduler
            early_stopping_patience: Patience for early stopping
            save_best: Whether to save best model
            gradient_accumulation_steps: Number of steps to accumulate gradients
            mixed_precision: Whether to use mixed precision training
            
        Returns:
            Training history
        """
        # Setup optimizer and scheduler
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
        
        # Setup mixed precision scaler if enabled
        scaler = None
        if mixed_precision and self.device.type == 'cuda':
            scaler = torch.cuda.amp.GradScaler()
        
        # Setup loss function
        from model import MultiLabelLoss
        criterion = MultiLabelLoss()
        
        # Setup early stopping
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion, epoch, 
                                                   gradient_accumulation_steps, scaler)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader, criterion, epoch)
            
            # Update learning rate
            scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            for attr_name, acc in train_acc.items():
                self.history['train_acc'][attr_name].append(acc)
            for attr_name, acc in val_acc.items():
                self.history['val_acc'][attr_name].append(acc)
            
            # Log epoch results
            epoch_time = time.time() - start_time
            print(f'\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s)')
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print('Train Acc:', {k: f'{v:.4f}' for k, v in train_acc.items()})
            print('Val Acc:', {k: f'{v:.4f}' for k, v in val_acc.items()})
            print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Save best model
            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, is_best=True)
                print(f'New best model saved! Val Loss: {val_loss:.4f}')
            
            # Check early stopping
            if early_stopping(val_loss, self.model):
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        # Save final model
        self.save_checkpoint(epoch, val_loss, is_best=False, is_final=True)
        
        return self.history
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False, 
                       is_final: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            val_loss: Validation loss
            is_best: Whether this is the best model
            is_final: Whether this is the final model
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }
        
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, 'best_model.pth'))
        elif is_final:
            torch.save(checkpoint, os.path.join(self.save_dir, 'final_model.pth'))
        else:
            torch.save(checkpoint, os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)
        return checkpoint
    
    def save_history(self, filename: str = 'training_history.json'):
        """Save training history to JSON file."""
        history_path = os.path.join(self.save_dir, filename)
        
        # Convert defaultdict to regular dict for JSON serialization
        serializable_history = {}
        for key, value in self.history.items():
            if isinstance(value, defaultdict):
                serializable_history[key] = dict(value)
            else:
                serializable_history[key] = value
        
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """
        Plot training curves.
        
        Args:
            save_path: Path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss curves
            axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
            axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Accuracy curves for each attribute
            for i, (attr_name, acc_history) in enumerate(self.history['val_acc'].items()):
                if i < 3:  # Plot up to 3 attributes
                    row, col = (i + 1) // 2, (i + 1) % 2
                    axes[row, col].plot(self.history['train_acc'][attr_name], 
                                      label=f'Train {attr_name}')
                    axes[row, col].plot(acc_history, label=f'Val {attr_name}')
                    axes[row, col].set_title(f'{attr_name.title()} Accuracy')
                    axes[row, col].set_xlabel('Epoch')
                    axes[row, col].set_ylabel('Accuracy')
                    axes[row, col].legend()
                    axes[row, col].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), 
                           dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Skipping plot generation.")


if __name__ == "__main__":
    # Test the trainer
    from model import create_model, MultiLabelLoss
    from data_loader import create_data_loaders
    
    # Create dummy data for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dummy model
    num_classes = {'make': 5, 'model': 10, 'steer_wheel': 2}
    model = create_model(backbone='resnet50', num_classes=num_classes)
    
    # Create trainer
    trainer = ModelTrainer(model, device, save_dir='test_checkpoints')
    
    print("Trainer created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
