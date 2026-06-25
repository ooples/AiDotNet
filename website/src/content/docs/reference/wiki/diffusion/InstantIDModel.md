---
title: "InstantIDModel<T>"
description: "InstantID model — zero-shot identity-preserving image generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Control`

InstantID model — zero-shot identity-preserving image generation.

## For Beginners

InstantID generates images of a specific person from one photo:

Key characteristics:

- Single reference image: no training or fine-tuning needed
- Face identity preservation: maintains facial features
- Text prompt control: change style, pose, expression
- Works with SDXL for high-quality output

How InstantID works:

1. Face encoder extracts identity embedding from reference image
2. IP-Adapter injects identity into cross-attention layers
3. IdentityNet provides spatial face guidance (landmarks)
4. Text prompt controls style, background, and expression

Use InstantID when you need:

- Personalized image generation from one photo
- Face-consistent character generation
- Style transfer while preserving identity

## How It Works

InstantID enables identity-preserving image generation from a single reference face image
without any fine-tuning. It combines a face encoder, IP-Adapter, and IdentityNet
to preserve facial identity while following text prompts.

Technical specifications:

- Architecture: SDXL + IP-Adapter + IdentityNet
- Face encoder: InsightFace (512-dim embedding)
- Base model: SDXL U-Net
- Control: face landmarks via IdentityNet
- Resolution: 1024x1024

Reference: Wang et al., "InstantID: Zero-shot Identity-Preserving Generation in Seconds", 2024

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `FaceEmbeddingDim` | Gets the face embedding dimension. |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |
| `GenerateFromText(String,String,Int32,Int32,Int32,Nullable<Double>,Nullable<Int32>)` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `SetParameters(Vector<>)` |  |

