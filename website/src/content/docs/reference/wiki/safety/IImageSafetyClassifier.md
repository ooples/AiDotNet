---
title: "IImageSafetyClassifier<T>"
description: "Interface for image safety classifiers that detect NSFW, violent, or otherwise harmful images."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Safety.Image`

Interface for image safety classifiers that detect NSFW, violent, or otherwise harmful images.

## For Beginners

An image safety classifier looks at an image and determines
whether it contains harmful content. Different implementations use different approaches —
CLIP embeddings, Vision Transformers, scene graph analysis, or an ensemble of all three.

## How It Works

Image safety classifiers analyze image tensors and assign per-category safety scores
for categories including sexual content, violence, self-harm, hate symbols, drugs,
child exploitation, shocking content, and dangerous activities.

**References:**

- UnsafeBench: 11 categories, GPT-4V achieves top F1 (2024, arxiv:2405.03486)
- USD: Scene-graph-based NSFW detection (USENIX Security 2025)
- Sensitive image classification via Vision Transformers (2024, arxiv:2412.16446)

## Methods

| Method | Summary |
|:-----|:--------|
| `GetCategoryScores(Tensor<>)` | Gets per-category safety scores for the given image. |

