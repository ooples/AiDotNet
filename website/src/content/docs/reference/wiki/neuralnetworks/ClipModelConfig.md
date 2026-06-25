---
title: "ClipModelConfig"
description: "Configuration for a CLIP model variant."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.NeuralNetworks`

Configuration for a CLIP model variant.

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` | The embedding dimension (e.g., 512 for ViT-B, 768 for ViT-L). |
| `ImageEncoderFile` | The filename of the image encoder ONNX model. |
| `ImageSize` | The expected image size (e.g., 224 or 336). |
| `MaxSequenceLength` | The maximum text sequence length (typically 77 for CLIP). |
| `TextEncoderFile` | The filename of the text encoder ONNX model. |

