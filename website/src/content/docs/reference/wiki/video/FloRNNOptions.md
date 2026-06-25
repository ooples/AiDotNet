---
title: "FloRNNOptions"
description: "Configuration options for FloRNN optical-flow-guided recurrent video denoising."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for FloRNN optical-flow-guided recurrent video denoising.

## For Beginners

FloRNN removes noise from video by tracking how objects move
between frames (optical flow) and using that to align information from previous frames.
This lets it average out random noise while keeping real details sharp.

## How It Works

FloRNN (Li et al., AAAI 2022) uses optical flow to guide recurrent denoising:

- Flow-guided alignment: warps previous hidden states using estimated optical flow before

feeding them to the recurrent unit, ensuring temporal features align with current frame

- Recurrent architecture: ConvLSTM/ConvGRU processes temporally aligned features,

accumulating clean signal over time while averaging out random noise

- Occlusion-aware gating: learned gates suppress features from occluded regions where

flow-based alignment is unreliable, preventing ghosting artifacts

- Multi-scale processing: operates at multiple spatial scales to handle both fine noise

patterns and large noisy regions

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FloRNNOptions` | Initializes a new instance with default values. |
| `FloRNNOptions(FloRNNOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `HiddenDim` | Gets or sets the hidden dimension for recurrent cells. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumFlowScales` | Gets or sets the number of flow estimation scales. |
| `NumRecurrentLayers` | Gets or sets the number of recurrent layers. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `Variant` | Gets or sets the model variant. |

