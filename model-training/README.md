# VisionGuard Vehicle Detection Model Training

This directory contains the training pipeline for YOLOv11-pose model optimized for Jetson Orin Nano deployment.

## Directory Structure
```
model-training/
├── data/                    # Dataset directory
│   ├── raw/                # Raw images and annotations
│   ├── processed/          # Processed and augmented dataset
│   └── splits/             # Train/val/test splits
├── models/                 # Model checkpoints and exports
│   ├── checkpoints/       # Training checkpoints
│   └── exported/          # Exported models (TensorRT, ONNX)
├── configs/               # Configuration files
├── scripts/              # Training and utility scripts
└── notebooks/           # Jupyter notebooks for analysis
```

## Setup Instructions

1. Environment Setup
```bash
# Create conda environment
conda create -n visionguard-train python=3.10
conda activate visionguard-train

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install additional dependencies
pip install -r requirements.txt
```

2. Dataset Preparation
```bash
# Download and prepare dataset
python scripts/prepare_dataset.py

# Generate data splits
python scripts/generate_splits.py
```

3. Training
```bash
# Start training
python scripts/train.py --config configs/yolov11_pose.yaml
```

4. Export
```bash
# Export to TensorRT
python scripts/export.py --format tensorrt --checkpoint models/checkpoints/best.pt
```

## Hardware Requirements

- NVIDIA Jetson Orin Nano
- 8GB+ RAM
- 100GB+ storage space for dataset and models

## Model Architecture

The YOLOv11-pose model has been optimized for vehicle pose estimation with the following modifications:
- Backbone: CSPDarknet optimized for Jetson hardware
- Custom neck architecture for improved feature fusion
- Modified head for vehicle-specific pose estimation
- TensorRT optimization for Jetson deployment

## Performance Metrics

Target metrics for Jetson Orin Nano:
- Inference speed: 30+ FPS
- mAP@0.5: 85%+
- Pose estimation accuracy: ±5° for orientation
