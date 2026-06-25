---
title: "JanusProOptions"
description: "Configuration options for Janus-Pro: scaled data and model with optimized training strategy."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Unified`

Configuration options for Janus-Pro: scaled data and model with optimized training strategy.

## For Beginners

These options configure the JanusPro model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `JanusProOptions(JanusProOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CfgScale` | Classifier-free guidance scale used during generation (Ho & Salimans 2022). |
| `CodebookEmbeddingDim` | Dimensionality of each VQ codebook entry's continuous embedding. |
| `EnableDecoupledEncoding` | Whether to keep understanding and generation vision paths fully decoupled (Janus paper §3.1). |
| `NumGenerationTokens` | Number of VQ tokens emitted by the generation path. |

