---
title: "QwenVL<T>"
description: "Qwen-VL: visual window attention, multi-resolution, bounding box output via cross-attention resampler."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

Qwen-VL: visual window attention, multi-resolution, bounding box output via cross-attention resampler.

## For Beginners

Qwen-VL from Alibaba is a versatile model that goes beyond
just describing images — it can locate objects by outputting bounding box coordinates.
It uses visual window attention (processing image regions locally for efficiency) and a
cross-attention resampler that compresses the large number of visual tokens into a fixed
smaller set before feeding them to the language model. It supports images at multiple
resolutions and can handle tasks like visual question answering, text reading (OCR),
and visual grounding (finding where objects are in an image). Default values follow the
original paper settings.

## How It Works

Qwen-VL (Bai et al., 2023) uses a ViT vision encoder with visual window attention and a
cross-attention resampler to compress visual features before feeding into the Qwen language
model. It supports multi-resolution input and can output bounding box coordinates for
visual grounding tasks.

**References:**

- Paper: "Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond" (Bai et al., 2023)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using Qwen-VL's window-attention + cross-attention resampler architecture. |
| `GetExtraTrainableLayers` |  |
| `PredictCore(Tensor<>)` | Vision-encoder-only forward: runs the patch-embedding + transformer stack in `Layers` and returns the resulting [B, S, VisionEmbeddingDim] embeddings. |

