---
title: "FlowLensOptions"
description: "Configuration options for the FlowLens video inpainting model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the FlowLens video inpainting model.

## For Beginners

FlowLens options configure the flow-guided video inpainting model.

## How It Works

**References:**

- Paper: "FlowLens: Seeing Beyond the FoV via Optical Flow Completion" (Xu et al., ECCV 2022)

FlowLens completes optical flow in masked regions first, then uses the completed flow
for high-quality temporal propagation followed by a refinement network, decoupling
motion estimation from pixel synthesis.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FlowLensOptions` | Initializes a new instance with default values. |
| `FlowLensOptions(FlowLensOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Dropout rate for regularization. |
| `LearningRate` | Learning rate for training. |
| `ModelPath` | Path to the ONNX model file for inference mode. |
| `NumFeatures` | Number of base feature channels. |
| `NumFlowIters` | Number of flow completion iterations for refining estimated flow in masked regions. |
| `NumLevels` | Number of encoder-decoder levels in the refinement network. |
| `NumResBlocks` | Number of residual blocks in the refinement branch. |
| `OnnxOptions` | ONNX runtime options for inference mode. |
| `Variant` | Model variant controlling capacity and speed trade-off. |

