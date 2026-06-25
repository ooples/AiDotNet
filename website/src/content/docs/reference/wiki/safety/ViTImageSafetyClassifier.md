---
title: "ViTImageSafetyClassifier<T>"
description: "Vision Transformer (ViT)-inspired image safety classifier using patch-based feature extraction and multi-head attention pooling for multi-label safety classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Image`

Vision Transformer (ViT)-inspired image safety classifier using patch-based feature extraction
and multi-head attention pooling for multi-label safety classification.

## For Beginners

Rather than looking at the whole image at once (like CLIP), this
classifier cuts the image into small squares (patches), analyzes each one separately, then
combines the results using attention — paying more attention to suspicious patches. This
approach catches localized unsafe content even in mostly-safe images.

## How It Works

Divides the image into fixed-size patches, computes feature embeddings per patch using
spatial statistics and color histograms, then aggregates with attention-weighted pooling.
The aggregated representation is classified against multiple safety categories using
per-category linear classifiers with learned biases.

**References:**

- Sensitive image classification via Vision Transformers (2024, arxiv:2412.16446)
- UnsafeBench: 11 categories, GPT-4V achieves top F1 (2024, arxiv:2405.03486)
- ShieldDiff: RL-based suppression of sexual content in diffusion (2024, arxiv:2410.05309)
- ViT: An Image is Worth 16x16 Words (Dosovitskiy et al., ICLR 2021)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ViTImageSafetyClassifier(Int32,Double)` | Initializes a new ViT-inspired image safety classifier. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateImage(Tensor<>)` |  |

