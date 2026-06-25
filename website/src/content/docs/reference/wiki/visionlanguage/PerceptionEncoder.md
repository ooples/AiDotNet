---
title: "PerceptionEncoder<T>"
description: "Meta's Perception Encoder for multimodal alignment tasks with global and dense features."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Encoders`

Meta's Perception Encoder for multimodal alignment tasks with global and dense features.

## For Beginners

Meta's Perception Encoder is a vision backbone that produces
both global features (a single vector summarizing the whole image) and dense spatial
features (one vector per image region). This makes it versatile — the global features
work for classification and retrieval, while the spatial features enable detection and
segmentation. Default values follow the original paper settings.

## How It Works

Perception Encoder (Meta, 2025) is a vision encoder combining contrastive learning with dense
prediction objectives. It produces both global features (via CLS token) and spatial features
(via patch tokens), making it suitable for classification, detection, and segmentation tasks
in multimodal systems.

**References:**

- Paper: "Perception Encoder: The best satisfies all" (Meta, 2025)

