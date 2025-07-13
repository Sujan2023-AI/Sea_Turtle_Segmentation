# Automated Sea Turtle Body Parts Segmentation

## Overview
Computer vision project for automated segmentation of sea turtle body parts (head, flippers, carapace) using deep learning models on the SeaTurtleID2022 dataset.

## Models Implemented
- **U-Net**: Custom architecture with encoder-decoder structure and skip connections
- **DeepLabV3**: ResNet-50 backbone with Atrous Spatial Pyramid Pooling (ASPP)
- **PSPNet**: Pyramid pooling module for multi-scale contextual information

## Dataset & Preprocessing
- **SeaTurtleID2022**: 2,000 images subset from 8,729 total images (13-year span)
- **Time-aware split**: Training (2010-2018), Validation (2019), Test (2020+)
- **Preprocessing**: 224x224 resize, data augmentation (flip, rotation, brightness)

## Key Results (Mean IoU)

| Model | Background | Head | Flipper | Carapace |
|-------|------------|------|---------|----------|
| U-Net | 0.9736 | 0.6783 | 0.4924 | 0.5309 |
| DeepLabV3 | 0.9796 | 0.7532 | 0.5069 | 0.5122 |
| **PSPNet** | **0.9880** | **0.8560** | **0.6938** | **0.7163** |

## Key Findings
- **PSPNet achieved best performance** across all body parts with highest IoU scores
- **Multi-scale context crucial** for accurate segmentation of challenging regions
- **Time-aware splitting** prevented data leakage and improved real-world applicability
- **Flipper and carapace segmentation** most challenging due to thin/low-contrast regions

## Technical Setup
- **Framework**: TensorFlow, Google Colab (A100 GPU)
- **Training**: 30-50 epochs, Adam optimizer, sparse categorical cross-entropy loss
- **Optimization**: Early stopping, learning rate scheduling

## Performance Highlights
PSPNet's superior performance attributed to:
- Pyramid pooling for multi-scale feature extraction
- Stable training convergence with minimal fluctuations
- Effective handling of challenging segmentation boundaries