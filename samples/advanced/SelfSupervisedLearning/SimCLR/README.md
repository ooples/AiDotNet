# SimCLR - Self-Supervised Contrastive Learning

This sample demonstrates how to use SimCLR (Simple Contrastive Learning of Visual Representations) for self-supervised pre-training with AiDotNet.

## Overview

SimCLR learns visual representations without labeled data by:
1. Creating augmented views of the same image
2. Training the model to recognize that augmentations of the same image are similar
3. Learning useful features that transfer to downstream tasks

## Prerequisites

- .NET 8.0 SDK or later
- AiDotNet NuGet package
- (Optional) NVIDIA GPU with CUDA for faster training

## Running the Sample

```bash
cd samples/advanced/SelfSupervisedLearning/SimCLR
dotnet run
```

## What This Sample Demonstrates

1. **Contrastive Learning Setup**: Configuring SimCLR with proper augmentations
2. **Projection Head**: Adding the projection MLP for contrastive learning
3. **NT-Xent Loss**: Using normalized temperature-scaled cross-entropy loss
4. **Feature Extraction**: Using pre-trained features for downstream tasks
5. **Transfer Learning**: Fine-tuning on labeled data after pre-training

## Key Concepts

### Data Augmentation Pipeline
SimCLR relies heavily on data augmentation:
- Random cropping and resizing
- Color jittering
- Gaussian blur
- Random horizontal flip

### Contrastive Loss (NT-Xent)
The model learns by maximizing agreement between differently augmented views of the same image.

### Two-Stage Training
1. **Pre-training**: Learn representations on unlabeled data
2. **Fine-tuning**: Train classifier on labeled data using learned features

## Code Structure

- `Program.cs` - Main entry point with SimCLR training pipeline
- Pre-training with contrastive learning
- Feature extraction from trained encoder
- Fine-tuning for classification

## Related Samples

- [BYOL](../BYOL/) - Bootstrap Your Own Latent
- [MoCo](../MoCo/) - Momentum Contrast
- [DINO](../DINO/) - Self-distillation with no labels

## Learn More

- [Self-Supervised Learning Guide](/docs/tutorials/self-supervised-learning/)
- [Transfer Learning](/docs/tutorials/transfer-learning/)
- [AiDotNet.SelfSupervisedLearning API Reference](/api/AiDotNet.SelfSupervisedLearning/)
