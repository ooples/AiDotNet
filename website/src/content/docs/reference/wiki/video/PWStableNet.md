---
title: "PWStableNet<T>"
description: "PWStableNet pixel-wise warping video stabilization with per-pixel flow fields."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Stabilization`

PWStableNet pixel-wise warping video stabilization with per-pixel flow fields.

## For Beginners

PWStableNet (Pixel-Wise Stable Network) stabilizes video with pixel-level precision, handling parallax and dynamic scenes better than methods that assume a single global motion.

## How It Works

**References:**

- Paper: "PWStableNet: Learning Pixel-Wise Warping Maps for Video Stabilization" (Zhao et al., IEEE TIP 2020)

PWStableNet predicts per-pixel warping maps instead of global homographies, enabling
more flexible stabilization that can handle parallax, rolling shutter distortion, and
depth-dependent motion, with coarse-to-fine refinement of the warp field.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PWStableNet(NeuralNetworkArchitecture<>,PWStableNetOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a PWStableNet model for native training and inference. |
| `PWStableNet(NeuralNetworkArchitecture<>,String,PWStableNetOptions)` | Creates a PWStableNet model for ONNX inference. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PostprocessOutput(Tensor<>)` |  |
| `PreprocessFrames(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Stabilize(Tensor<>)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

