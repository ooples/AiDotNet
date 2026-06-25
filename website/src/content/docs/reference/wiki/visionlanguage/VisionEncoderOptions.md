---
title: "VisionEncoderOptions"
description: "Base configuration options for standalone vision encoder models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Encoders`

Base configuration options for standalone vision encoder models.

## How It Works

Vision encoders extract feature representations from images without a paired text encoder.
They are used as backbones for downstream VLMs, classification, detection, and segmentation tasks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VisionEncoderOptions` | Initializes a new instance with default values. |
| `VisionEncoderOptions(VisionEncoderOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `EmbeddingDim` | Gets or sets the vision encoder embedding dimension. |
| `FfnMultiplier` | Gets or sets the feed-forward hidden dimension multiplier. |
| `ImageMean` | Gets or sets the per-channel mean for image normalization. |
| `ImageSize` | Gets or sets the input image size (height = width). |
| `ImageStd` | Gets or sets the per-channel standard deviation for image normalization. |
| `LearningRate` | Gets or sets the initial learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of transformer layers. |
| `OnnxOptions` | Gets or sets ONNX runtime options. |
| `PatchSize` | Gets or sets the ViT patch size in pixels. |
| `WeightDecay` | Gets or sets the weight decay for regularization. |

