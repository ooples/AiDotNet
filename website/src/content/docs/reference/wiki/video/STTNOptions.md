---
title: "STTNOptions"
description: "Configuration options for the STTN (Spatial-Temporal Transformer Network) video inpainting model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the STTN (Spatial-Temporal Transformer Network) video inpainting model.

## For Beginners

STTN options configure the spatial-temporal transformer inpainting network.

## How It Works

**References:**

- Paper: "Learning Joint Spatial-Temporal Transformations for Video Inpainting" (Zeng et al., ECCV 2020)

STTN uses multi-scale spatial-temporal transformers to simultaneously search for and
attend to relevant patches across space and time, filling masked regions with content
from visible areas in the same and other frames.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `STTNOptions` | Initializes a new instance with default values. |
| `STTNOptions(STTNOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Dropout rate for regularization. |
| `LearningRate` | Learning rate for training. |
| `ModelPath` | Path to the ONNX model file for inference mode. |
| `NumFeatures` | Number of base feature channels. |
| `NumHeads` | Number of attention heads for multi-head spatial-temporal attention. |
| `NumScales` | Number of scales for multi-scale patch matching. |
| `NumTransformerLayers` | Number of spatial-temporal transformer layers. |
| `OnnxOptions` | ONNX runtime options for inference mode. |
| `Variant` | Model variant controlling capacity and speed trade-off. |

