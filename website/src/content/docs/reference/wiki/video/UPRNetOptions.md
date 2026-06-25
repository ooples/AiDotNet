---
title: "UPRNetOptions"
description: "Configuration options for UPR-Net unified pyramid recurrent network."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for UPR-Net unified pyramid recurrent network.

## For Beginners

UPR-Net combines motion estimation and frame creation into a single
efficient network that processes images at multiple scales. At each scale, it repeatedly
refines its predictions until they're good enough, like iterating on a drawing.

## How It Works

UPR-Net (Ma et al., 2023) uses a unified pyramid recurrent architecture:

- Unified pyramid: a single encoder-decoder pyramid that performs both optical flow estimation

and frame synthesis in one pass, avoiding redundant feature computation and sharing
multi-scale representations between the two tasks

- Recurrent refinement: at each pyramid level, a ConvLSTM recurrently refines flow and frame

predictions, iterating until convergence rather than using a fixed number of steps

- Bidirectional estimation: simultaneously estimates forward and backward flows with shared

weights, using consistency checks between the two directions to detect occlusions

- Lightweight design: the unified architecture removes the need for separate flow and

synthesis networks, reducing parameters significantly while maintaining quality

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UPRNetOptions` | Initializes a new instance with default values. |
| `UPRNetOptions(UPRNetOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `LSTMHiddenDim` | Gets or sets the ConvLSTM hidden dimension. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumPyramidLevels` | Gets or sets the number of pyramid levels. |
| `NumRecurrentIters` | Gets or sets the number of recurrent refinement iterations per level. |
| `NumResBlocks` | Gets or sets the number of residual blocks per pyramid level. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `Variant` | Gets or sets the model variant. |

