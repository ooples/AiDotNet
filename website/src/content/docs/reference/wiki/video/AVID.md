---
title: "AVID<T>"
description: "AVID diffusion-based video inpainting supporting arbitrary-length videos."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Inpainting`

AVID diffusion-based video inpainting supporting arbitrary-length videos.

## For Beginners

AVID (Adaptive Video Inpainting via Diffusion) fills in missing or damaged regions of video using diffusion models. It adaptively propagates content from neighboring frames and regions.

## How It Works

**References:**

- Paper: "AVID: Any-Length Video Inpainting with Diffusion Model" (Zhang et al., CVPR 2024)

AVID uses a diffusion U-Net with temporal attention to iteratively denoise masked video regions,
processing long videos through an autoregressive temporal pipeline with overlapping windows
that maintains temporal consistency across the full sequence.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AVID(NeuralNetworkArchitecture<>,AVIDOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an AVID model for native training and inference. |
| `AVID(NeuralNetworkArchitecture<>,String,AVIDOptions)` | Creates an AVID model for ONNX inference. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `Inpaint(Tensor<>,Tensor<>)` |  |
| `PostprocessOutput(Tensor<>)` |  |
| `PreprocessFrames(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

