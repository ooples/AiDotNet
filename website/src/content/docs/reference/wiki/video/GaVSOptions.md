---
title: "GaVSOptions"
description: "Configuration options for the GaVS (Gaze-aware Video Stabilization) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the GaVS (Gaze-aware Video Stabilization) model.

## For Beginners

GaVS options configure the adversarial video stabilization model.

## How It Works

**References:**

- Paper: "Gaze-aware Video Stabilization" (2023)

GaVS incorporates gaze prediction to stabilize video while preserving the viewer's
region of interest, weighting stabilization strength based on visual saliency.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GaVSOptions` | Initializes a new instance with default values. |
| `GaVSOptions(GaVSOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Dropout rate for regularization. |
| `GazeHiddenDim` | Hidden dimension for the gaze prediction branch. |
| `LearningRate` | Learning rate for training. |
| `ModelPath` | Path to the ONNX model file for inference mode. |
| `NumFeatures` | Number of base feature channels. |
| `NumGazeHeads` | Number of gaze prediction heads for multi-scale saliency estimation. |
| `OnnxOptions` | ONNX runtime options for inference mode. |
| `SmoothingWindow` | Smoothing window size for trajectory filtering. |
| `Variant` | Model variant controlling capacity and speed trade-off. |

