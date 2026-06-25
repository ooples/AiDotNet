---
title: "InternViT<T>"
description: "InternViT, a 6B-parameter ViT used as the vision encoder in the InternVL series."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Encoders`

InternViT, a 6B-parameter ViT used as the vision encoder in the InternVL series.

## For Beginners

InternViT is a 6 billion parameter Vision Transformer designed
as the vision backbone for the InternVL series of multimodal models. It handles high-
resolution images by dynamically tiling them and using pixel shuffle to reduce the number
of visual tokens while preserving spatial detail. It is progressively aligned with large
language models for visual-linguistic tasks. Default values follow the original paper
settings.

## How It Works

InternViT (Chen et al., 2024) is a large-scale ViT designed for progressive alignment with LLMs.
It uses dynamic resolution via image tiling and pixel shuffle downsampling to reduce token counts
while preserving spatial detail, making it efficient for high-resolution images.

**References:**

- Paper: "InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks" (Chen et al., 2024)

