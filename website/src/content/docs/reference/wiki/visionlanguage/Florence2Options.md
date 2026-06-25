---
title: "Florence2Options"
description: "Configuration options for Florence-2, Microsoft's unified vision foundation model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Encoders`

Configuration options for Florence-2, Microsoft's unified vision foundation model.

## How It Works

Florence-2 (Xiao et al., 2024) is a lightweight sequence-to-sequence vision foundation model
(0.23B-0.77B) that handles captioning, object detection, grounding, OCR, and segmentation through
a unified prompt-based approach. It uses DaViT as the vision encoder and a multi-task decoder.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Florence2Options(Florence2Options)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DecoderEmbeddingDim` | Gets or sets the decoder embedding dimension. |
| `MaxOutputTokens` | Gets or sets the maximum number of output tokens for the text decoder. |
| `ModelSize` | Gets or sets the model size variant. |
| `NumDecoderHeads` | Gets or sets the number of decoder attention heads. |
| `NumDecoderLayers` | Gets or sets the number of decoder layers. |
| `UseDaViT` | Gets or sets whether to use the DaViT (Dual Attention Vision Transformer) backbone. |
| `VocabSize` | Gets or sets the vocabulary size for the text decoder. |

