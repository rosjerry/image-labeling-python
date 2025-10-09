#!/usr/bin/env python
"""
Setup script for car_classifier package.

Install the package in development mode:
    pip install -e .

Install the package:
    pip install .
"""

from setuptools import setup, find_packages
import os


# Read the long description from README
def read_long_description():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "Multi-label car classification package"


# Read requirements
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            return [
                line.strip() for line in fh if line.strip() and not line.startswith("#")
            ]
    except FileNotFoundError:
        return []


setup(
    name="car-classifier",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Multi-label car classification using PyTorch and Label Studio",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/image-labeling-python",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "car-classifier-train=examples.train:main",
            "car-classifier-inference=car_classifier.inference:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
