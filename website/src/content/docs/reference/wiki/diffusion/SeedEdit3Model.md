---
title: "SeedEdit3Model<T>"
description: "SeedEdit 3 model for high-fidelity instruction-based editing with structure preservation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ImageEditing`

SeedEdit 3 model for high-fidelity instruction-based editing with structure preservation.

## For Beginners

SeedEdit 3 is great at making changes while keeping the image
layout the same. If you ask it to "change the dog to a cat," the cat will be the same
size and position as the dog was. This is important for consistent editing.

## How It Works

SeedEdit 3 focuses on preserving the spatial structure and layout of the original image
while applying requested edits. Uses a structure-aware loss during training to maintain
object positions, sizes, and relationships even during significant content changes.
Built on a DiT backbone with rectified flow training.

Technical specifications:

- Architecture: DiT-based with structure-aware conditioning
- Latent space: 16 channels with rectified flow
- Structure preservation: Layout-aware loss + spatial conditioning
- Guidance: 5.0 (balances edit strength vs structure fidelity)

Reference: ByteDance, "SEED-Edit", 2024

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `SetParameters(Vector<>)` |  |

