---
title: "VideoChat2<T>"
description: "VideoChat2: progressive video training with diverse data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.VideoLanguage`

VideoChat2: progressive video training with diverse data.

## For Beginners

VideoChat2 is a video-language model trained progressively on
diverse video-text data for video conversation. Default values follow the original paper
settings.

## How It Works

VideoChat2 (Shanghai AI Lab, 2023) uses progressive video training with diverse multi-modal
data. It trains through multiple stages starting from image-text alignment, then image
instruction tuning, and finally video instruction tuning on diverse video datasets, building
strong video conversation capabilities incrementally.

**References:**

- Paper: "MVBench: A Comprehensive Multi-modal Video Understanding Benchmark" (Shanghai AI Lab, 2023)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates from a single image using VideoChat2's Q-Former visual-text cross-attention. |
| `GenerateFromVideo(IReadOnlyList<Tensor<>>,String)` | Generates output from video frames using VideoChat2's Q-Former temporal aggregation. |

