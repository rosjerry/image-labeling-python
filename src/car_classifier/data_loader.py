import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Any
import re


class CarDataset(Dataset):
    """
    Dataset class for car image classification with multi-label annotations.
    Handles Label Studio JSON format and multi-label classification.
    """

    def __init__(
        self,
        json_path: str,
        image_dir: str,
        transform=None,
        label_mapping: Dict[str, Dict[str, int]] = None,
    ):
        """
        Initialize the dataset.

        Args:
            json_path: Path to Label Studio JSON file
            image_dir: Directory containing images
            transform: Image transformations
            label_mapping: Optional pre-computed label mappings
        """
        self.json_path = json_path
        self.image_dir = image_dir
        self.transform = transform

        # Load and parse JSON data
        self.data = self._load_json_data()

        # Extract unique labels and create mappings
        if label_mapping is None:
            self.label_mapping = self._create_label_mappings()
        else:
            self.label_mapping = label_mapping

        # Create samples list
        self.samples = self._create_samples()

    def _load_json_data(self) -> List[Dict]:
        """Load and parse Label Studio JSON data."""
        with open(self.json_path, "r") as f:
            return json.load(f)

    def _create_label_mappings(self) -> Dict[str, Dict[str, int]]:
        """Create label mappings for each attribute."""
        label_sets = {"make": set(), "model": set(), "steer_wheel": set()}

        # Extract all unique labels
        for item in self.data:
            if "annotations" in item and len(item["annotations"]) > 0:
                annotation = item["annotations"][0]  # Take first annotation
                if "result" in annotation:
                    for result in annotation["result"]:
                        if result["type"] == "choices" and "value" in result:
                            choices = result["value"].get("choices", [])
                            if choices:
                                attr_name = result["from_name"]
                                if attr_name in label_sets:
                                    label_sets[attr_name].add(choices[0])

        # Create mappings
        mappings = {}
        for attr_name, labels in label_sets.items():
            sorted_labels = sorted(list(labels))
            mappings[attr_name] = {
                label: idx for idx, label in enumerate(sorted_labels)
            }

        return mappings

    def _create_samples(self) -> List[Dict]:
        """Create list of samples with image paths and labels."""
        samples = []

        for item in self.data:
            if "annotations" in item and len(item["annotations"]) > 0:
                annotation = item["annotations"][0]
                if "result" in annotation:
                    # Extract image path
                    image_path = self._extract_image_path(item)
                    if image_path and os.path.exists(image_path):
                        # Extract labels
                        labels = self._extract_labels(annotation["result"])
                        if labels:
                            samples.append({"image_path": image_path, "labels": labels})

        return samples

    def _extract_image_path(self, item: Dict) -> str:
        """Extract image path from data item."""
        if "data" in item and "image" in item["data"]:
            image_url = item["data"]["image"]
            # Extract filename from URL path
            # Format: /data/local-files/?d=toyota/rav4/45413425/45413425_Image_1.jpg
            match = re.search(r"/([^/]+\.jpg)$", image_url)
            if match:
                filename = match.group(1)

                # Try multiple possible locations for the image
                possible_paths = [
                    # Direct path in image_dir
                    os.path.join(self.image_dir, filename),
                    # Path with subdirectory structure
                    os.path.join(
                        self.image_dir,
                        "toyota",
                        "rav4",
                        filename.split("_")[0],
                        filename,
                    ),
                    # Flat structure
                    os.path.join(self.image_dir, filename),
                ]

                # Return the first path that exists
                for path in possible_paths:
                    if os.path.exists(path):
                        return path

                # If no path exists, return the first one (will be handled by error checking)
                return possible_paths[0]
        return None

    def _extract_labels(self, results: List[Dict]) -> Dict[str, int]:
        """Extract labels from annotation results."""
        labels = {}

        for result in results:
            if result["type"] == "choices" and "value" in result:
                choices = result["value"].get("choices", [])
                if choices:
                    attr_name = result["from_name"]
                    if attr_name in self.label_mapping:
                        label_value = choices[0]
                        if label_value in self.label_mapping[attr_name]:
                            labels[attr_name] = self.label_mapping[attr_name][
                                label_value
                            ]

        return labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get item by index."""
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample["image_path"]).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Create label tensors
        labels = {}
        for attr_name in ["make", "model", "steer_wheel"]:
            if attr_name in sample["labels"]:
                labels[attr_name] = torch.tensor(
                    sample["labels"][attr_name], dtype=torch.long
                )
            else:
                # Handle missing labels (shouldn't happen with proper data)
                labels[attr_name] = torch.tensor(0, dtype=torch.long)

        return image, labels

    def get_num_classes(self) -> Dict[str, int]:
        """Get number of classes for each attribute."""
        return {
            attr_name: len(mapping) for attr_name, mapping in self.label_mapping.items()
        }

    def get_label_mapping(self) -> Dict[str, Dict[str, int]]:
        """Get label mappings."""
        return self.label_mapping


def get_data_transforms(
    image_size: int = 224,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get training and validation data transforms.

    Args:
        image_size: Target image size

    Returns:
        Tuple of (train_transform, val_transform)
    """
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, val_transform


def create_data_loaders(
    json_path: str,
    image_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    val_split: float = 0.2,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = False,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """
    Create training and validation data loaders.

    Args:
        json_path: Path to Label Studio JSON file
        image_dir: Directory containing images
        batch_size: Batch size for data loaders
        image_size: Target image size
        val_split: Validation split ratio
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        persistent_workers: Whether to keep workers alive between epochs

    Returns:
        Tuple of (train_loader, val_loader, num_classes)
    """
    # Get transforms
    train_transform, val_transform = get_data_transforms(image_size)

    # Create full dataset to get label mappings
    full_dataset = CarDataset(json_path, image_dir, transform=None)
    num_classes = full_dataset.get_num_classes()
    label_mapping = full_dataset.get_label_mapping()

    # Split dataset
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size

    # Create indices for train/val split
    indices = torch.randperm(dataset_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create train dataset
    train_dataset = CarDataset(
        json_path, image_dir, transform=train_transform, label_mapping=label_mapping
    )
    train_dataset.samples = [full_dataset.samples[i] for i in train_indices]

    # Create validation dataset
    val_dataset = CarDataset(
        json_path, image_dir, transform=val_transform, label_mapping=label_mapping
    )
    val_dataset.samples = [full_dataset.samples[i] for i in val_indices]

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )

    return train_loader, val_loader, num_classes


if __name__ == "__main__":
    # Test the data loader
    json_path = "project-1-at-2025-10-05-21-43-98b8ac33.json"
    image_dir = "images"  # Adjust this path as needed

    if os.path.exists(json_path):
        train_loader, val_loader, num_classes = create_data_loaders(
            json_path, image_dir, batch_size=4, image_size=224
        )

        print(f"Number of classes: {num_classes}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")

        # Test a batch
        for images, labels in train_loader:
            print(f"Image shape: {images.shape}")
            print(f"Labels: {labels}")
            break
    else:
        print(f"JSON file not found: {json_path}")
