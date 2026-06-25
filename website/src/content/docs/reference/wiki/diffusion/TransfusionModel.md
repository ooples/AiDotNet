---
title: "TransfusionModel<T>"
description: "Transfusion model combining autoregressive language modeling with diffusion generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.FastGeneration`

Transfusion model combining autoregressive language modeling with diffusion generation.

## For Beginners

Most image generators have a separate text model and image model.
Transfusion combines both into one unified transformer — it reads and writes text
autoregressively (word by word) and generates images through diffusion (noise removal),
all in one model. This enables natural interleaving of text and images.

## How It Works

Transfusion trains a single transformer to handle both text (autoregressive) and images
(diffusion) within the same sequence. Text tokens are modeled autoregressively while
image patches use a diffusion loss, enabling native multimodal generation without
separate text and image models.

Reference: Zhou et al., "Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model", 2024

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

