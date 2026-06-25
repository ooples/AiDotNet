---
title: "IVisionLanguageFusionModel<T>"
description: "Interface for vision-language fusion models that combine image and text features for tasks like VQA, image-text matching, and cross-modal retrieval."
section: "API Reference"
---

`Interfaces` · `AiDotNet.VisionLanguage.Interfaces`

Interface for vision-language fusion models that combine image and text features
for tasks like VQA, image-text matching, and cross-modal retrieval.

## How It Works

Unlike contrastive models that keep vision and text separate, fusion models combine
modalities internally via co-attention, cross-attention, or single-stream architectures.
They support tasks including:

- Visual Question Answering (VQA) - answering questions about images
- Image-Text Matching (ITM) - determining if image and text match
- Cross-Modal Retrieval - finding relevant images for text or vice versa
- Visual Entailment - reasoning about image-text relationships

## Properties

| Property | Summary |
|:-----|:--------|
| `FusionEmbeddingDim` | Gets the fusion embedding dimension. |
| `MaxSequenceLength` | Gets the maximum text sequence length. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeMatchingScore(Tensor<>,String)` | Computes an image-text matching score indicating how well the pair matches. |
| `FuseImageText(Tensor<>,String)` | Computes a fused representation of an image-text pair. |

