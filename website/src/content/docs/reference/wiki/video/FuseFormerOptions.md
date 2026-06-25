---
title: "FuseFormerOptions"
description: "Configuration options for the FuseFormer video inpainting model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the FuseFormer video inpainting model.

## For Beginners

FuseFormer options configure the transformer video inpainting model.

## How It Works

**References:**

- Paper: "FuseFormer: Fusing Fine-Grained Information in Transformers for Video Inpainting" (Liu et al., ICCV 2021)

FuseFormer uses a transformer architecture with soft split and composition operations
to fuse fine-grained spatial-temporal information at multiple scales for high-quality
video inpainting with better detail preservation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FuseFormerOptions` | Initializes a new instance with default values. |
| `FuseFormerOptions(FuseFormerOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Dropout rate for regularization. |
| `LearningRate` | Learning rate for training. |
| `ModelPath` | Path to the ONNX model file for inference mode. |
| `NumFeatures` | Number of base feature channels. |
| `NumHeads` | Number of attention heads in the multi-head attention modules. |
| `NumTransformerLayers` | Number of transformer layers for temporal fusion. |
| `OnnxOptions` | ONNX runtime options for inference mode. |
| `PatchSize` | Patch size for the soft split operation that divides features into overlapping patches. |
| `Variant` | Model variant controlling capacity and speed trade-off. |

