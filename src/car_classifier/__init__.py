"""
Car Classifier Package

A multi-label car classification package for training and inference.
Predicts car attributes like make, model, and steering wheel position.
"""

__version__ = "1.0.0"

from .model import (
    create_model,
    MultiLabelCarClassifier,
    MultiLabelLoss,
    count_parameters,
)
from .trainer import ModelTrainer
from .data_loader import CarDataset, create_data_loaders, get_data_transforms

__all__ = [
    "create_model",
    "MultiLabelCarClassifier",
    "MultiLabelLoss",
    "count_parameters",
    "ModelTrainer",
    "CarDataset",
    "create_data_loaders",
    "get_data_transforms",
]
