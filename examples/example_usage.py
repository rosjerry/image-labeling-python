#!/usr/bin/env python3
"""
Example usage of the multi-label car classification pipeline.
This script demonstrates how to use the training pipeline with your data.
"""

import os
import json
from car_classifier.data_loader import create_data_loaders
from car_classifier.model import create_model
from car_classifier.trainer import ModelTrainer


def example_training():
    """Example of how to train the model."""
    print("Example: Training Multi-Label Car Classifier")
    print("="*50)
    
    # Configuration
    json_path = "project-1-at-2025-10-05-21-43-98b8ac33.json"  # Your Label Studio JSON
    image_dir = "images"  # Directory containing your images
    save_dir = "example_checkpoints"
    
    # Check if files exist
    if not os.path.exists(json_path):
        print(f"❌ JSON file not found: {json_path}")
        print("Please update the json_path variable with your Label Studio JSON file.")
        return
    
    if not os.path.exists(image_dir):
        print(f"❌ Image directory not found: {image_dir}")
        print("Please update the image_dir variable with your image directory.")
        return
    
    print(f"✓ JSON file found: {json_path}")
    print(f"✓ Image directory found: {image_dir}")
    
    # Create data loaders
    print("\n1. Creating data loaders...")
    train_loader, val_loader, num_classes = create_data_loaders(
        json_path=json_path,
        image_dir=image_dir,
        batch_size=16,  # Adjust based on your GPU memory
        image_size=224,
        val_split=0.2,
        num_workers=4
    )
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    print(f"✓ Number of classes: {num_classes}")
    
    # Create model
    print("\n2. Creating model...")
    model = create_model(
        backbone='resnet50',  # Try 'efficientnet_b0' for faster training
        num_classes=num_classes,
        pretrained=True,
        dropout_rate=0.5
    )
    
    print(f"✓ Model created with backbone: {model.backbone_name}")
    
    # Create trainer
    print("\n3. Setting up trainer...")
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = ModelTrainer(model, device, save_dir=save_dir)
    
    print(f"✓ Trainer created on device: {device}")
    
    # Train model
    print("\n4. Starting training...")
    print("This will train for a few epochs as an example.")
    print("For full training, increase the epochs parameter.")
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=5,  # Increase for full training
        learning_rate=1e-4,
        weight_decay=1e-4,
        scheduler_step=10,
        scheduler_gamma=0.1,
        early_stopping_patience=5,
        save_best=True
    )
    
    print("\n✓ Training completed!")
    print(f"✓ Best model saved to: {save_dir}/best_model.pth")
    print(f"✓ Training history saved to: {save_dir}/training_history.json")


def example_inference():
    """Example of how to use the trained model for inference."""
    print("\n" + "="*50)
    print("Example: Using Trained Model for Inference")
    print("="*50)
    
    # This would be used after training
    model_path = "example_checkpoints/best_model.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please train a model first using the example_training() function.")
        return
    
    print("To use the trained model for inference:")
    print("\n1. Single image prediction:")
    print("python inference.py --model_path example_checkpoints/best_model.pth \\")
    print("                    --label_mapping example_checkpoints/label_mapping.json \\")
    print("                    --image_path your_test_image.jpg")
    
    print("\n2. Batch prediction:")
    print("python inference.py --model_path example_checkpoints/best_model.pth \\")
    print("                    --label_mapping example_checkpoints/label_mapping.json \\")
    print("                    --image_dir test_images/ \\")
    print("                    --output_path predictions.json")


def example_command_line():
    """Example of command line usage."""
    print("\n" + "="*50)
    print("Example: Command Line Usage")
    print("="*50)
    
    print("1. Basic training:")
    print("python main.py --json_path project-1-at-2025-10-05-21-43-98b8ac33.json \\")
    print("                --image_dir images/ \\")
    print("                --backbone resnet50 \\")
    print("                --epochs 50 \\")
    print("                --batch_size 32")
    
    print("\n2. Advanced training with custom settings:")
    print("python main.py --json_path project-1-at-2025-10-05-21-43-98b8ac33.json \\")
    print("                --image_dir images/ \\")
    print("                --backbone efficientnet_b0 \\")
    print("                --epochs 100 \\")
    print("                --batch_size 16 \\")
    print("                --learning_rate 1e-4 \\")
    print("                --weight_decay 1e-4 \\")
    print("                --val_split 0.2 \\")
    print("                --save_dir my_checkpoints \\")
    print("                --device cuda")
    
    print("\n3. Training with frozen backbone (faster):")
    print("python main.py --json_path project-1-at-2025-10-05-21-43-98b8ac33.json \\")
    print("                --image_dir images/ \\")
    print("                --freeze_backbone \\")
    print("                --epochs 20 \\")
    print("                --learning_rate 1e-3")
    
    print("\n4. Inference:")
    print("python inference.py --model_path checkpoints/best_model.pth \\")
    print("                    --label_mapping checkpoints/label_mapping.json \\")
    print("                    --image_path test_image.jpg")


def main():
    """Main example function."""
    print("Multi-Label Car Classification Pipeline - Usage Examples")
    print("="*60)
    
    # Show different usage examples
    example_command_line()
    example_inference()
    
    # Uncomment the line below to run actual training (requires your data)
    # example_training()
    
    print("\n" + "="*60)
    print("Next Steps:")
    print("1. Prepare your Label Studio JSON file")
    print("2. Organize your images in a directory")
    print("3. Update the paths in this script or use command line arguments")
    print("4. Run: python main.py --json_path your_data.json --image_dir your_images/")
    print("="*60)


if __name__ == "__main__":
    main()
