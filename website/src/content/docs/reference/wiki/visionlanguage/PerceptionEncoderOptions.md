---
title: "PerceptionEncoderOptions"
description: "Configuration options for Meta's Perception Encoder vision foundation model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Encoders`

Configuration options for Meta's Perception Encoder vision foundation model.

## How It Works

Perception Encoder (Meta, 2025) is a vision encoder designed for multimodal alignment tasks.
It combines contrastive learning with dense prediction objectives to produce features suitable
for both global understanding and local spatial reasoning.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PerceptionEncoderOptions(PerceptionEncoderOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlignmentProjectionDim` | Gets or sets the alignment projection dimension. |
| `DenseFeatureStride` | Gets or sets the dense feature output stride. |
| `GlobalPooling` | Gets or sets the global feature pooling strategy. |
| `UseDenseFeatures` | Gets or sets whether to use dense prediction features alongside global features. |

