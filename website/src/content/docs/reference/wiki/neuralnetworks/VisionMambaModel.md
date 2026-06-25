---
title: "VisionMambaModel<T>"
description: "Implements the Vision Mamba (Vim) model: PatchEmbed + scan pattern + bidirectional Mamba + classifier."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Implements the Vision Mamba (Vim) model: PatchEmbed + scan pattern + bidirectional Mamba + classifier.

## For Beginners

This model classifies images using Mamba instead of attention.
It cuts the image into small patches, reads them in a specific order (the scan pattern),
processes with Mamba blocks (fast sequence processing), and outputs class probabilities.
Vision Mamba needs O(n) computation vs O(n^2) for Transformers, making it faster for
high-resolution images like medical imaging and satellite photos.

## How It Works

Vision Mamba adapts the Mamba SSM architecture for image classification by:

1. Dividing the image into non-overlapping patches (like ViT)
2. Projecting patches to an embedding dimension
3. Adding learnable positional embeddings
4. Scanning patches through Mamba blocks using configurable spatial scan patterns
5. Pooling the output and projecting to class logits

Different scan patterns capture different spatial relationships:

- `Bidirectional`: forward + reverse (Vim paper)
- `CrossScan`: 4 directional scans (VMamba)
- `Continuous`: zigzag preserving locality (PlainMamba)

**References:**

- Zhu et al., "Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model", 2024
- Liu et al., "VMamba: Visual State Space Model", 2024
- Yang et al., "PlainMamba: Improving Non-Hierarchical Mamba in Visual Recognition", 2024

## Properties

| Property | Summary |
|:-----|:--------|
| `ImageHeight` | Gets the image height. |
| `ImageWidth` | Gets the image width. |
| `ModelDimension` | Gets the model dimension (d_model). |
| `NumClasses` | Gets the number of output classes. |
| `NumLayers` | Gets the number of Mamba layers. |
| `NumPatches` | Gets the total number of patches. |
| `PatchSize` | Gets the patch size. |
| `ScanPattern` | Gets the scan pattern used. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ForwardForTraining(Tensor<>)` | ForwardForTraining MUST execute the same pipeline as Predict (patch embed → positional → scan pattern → Mamba blocks → pool → norm → classifier). |
| `GetOptions` |  |

