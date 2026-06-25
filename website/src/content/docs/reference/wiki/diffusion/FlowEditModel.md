---
title: "FlowEditModel<T>"
description: "FlowEdit model for image editing via rectified flow inversion and re-generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ImageEditing`

FlowEdit model for image editing via rectified flow inversion and re-generation.

## For Beginners

FlowEdit works by "un-generating" your image back to noise,
then re-generating it with a different text prompt. Because it follows a precise
mathematical path (rectified flow), it preserves the original image structure while
making the changes you describe.

## How It Works

FlowEdit performs image editing by first inverting the input image to its noise
representation using rectified flow ODE inversion, then re-generating with modified
text conditioning. This approach preserves image structure while enabling text-guided edits.

Technical specifications:

- Base model: FLUX.1 dev (hybrid MMDiT, 3072 hidden, 19+38 layers)
- Editing method: Flow ODE inversion + re-generation
- No explicit inversion network needed — uses ODE solver directly
- Supports partial inversion for structure preservation control
- Text encoders: CLIP ViT-L/14 + T5-XXL (inherited from FLUX)

Reference: Kulikov et al., "FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models", 2024

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

