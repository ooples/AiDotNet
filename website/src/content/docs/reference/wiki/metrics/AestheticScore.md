---
title: "AestheticScore<T>"
description: "Aesthetic Score metric using CLIP for evaluating image aesthetics."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Metrics`

Aesthetic Score metric using CLIP for evaluating image aesthetics.

## How It Works

Aesthetic Score uses CLIP embeddings trained on aesthetic preference data to predict
how visually appealing an image is. This is commonly used in image generation to
filter or rank outputs by aesthetic quality.

Typical aesthetic scores (1-10 scale):

- >7: High aesthetic quality
- 5-7: Average aesthetic quality
- <5: Low aesthetic quality

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AestheticScore(IMultimodalEmbedding<>,Vector<>,String[],String[])` | Initializes a new instance of AestheticScore calculator. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Double[])` | Computes aesthetic score for an image. |
| `ComputeBatch(IList<Double[]>)` | Computes aesthetic scores for a batch of images. |
| `ComputeWithTrainedWeights(Double[])` | Computes aesthetic score using trained weights. |
| `ComputeZeroShot(Double[])` | Computes aesthetic score using zero-shot comparison with aesthetic prompts. |
| `RankByAesthetic(IList<Double[]>)` | Ranks images by aesthetic score. |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultNegativePrompts` | Default negative aesthetic prompts used for zero-shot aesthetic scoring. |
| `DefaultPositivePrompts` | Default positive aesthetic prompts used for zero-shot aesthetic scoring. |

