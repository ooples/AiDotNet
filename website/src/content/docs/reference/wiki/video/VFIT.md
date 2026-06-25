---
title: "VFIT<T>"
description: "VFIT video frame interpolation transformer with multi-frame spatial-temporal attention."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.FrameInterpolation`

VFIT video frame interpolation transformer with multi-frame spatial-temporal attention.

## For Beginners

VFIT uses more than just two frames to create the in-between frame.
By looking at a wider window of frames (2 before and 2 after), it can better understand
complex motions like acceleration, deceleration, and periodic movements.

## How It Works

**References:**

- Paper: "Video Frame Interpolation Transformer" (Shi et al., CVPR 2022)

VFIT uses vision transformers for multi-frame interpolation with key innovations:

- Multi-frame input: takes multiple input frames (typically 4: two before and two after the

target) to provide richer temporal context than 2-frame methods

- Temporal transformer: applies temporal self-attention across the multiple input frames,

learning long-range temporal dependencies and motion patterns that span multiple frames

- Spatial-temporal factorization: factorizes the full 3D attention into separate spatial

(within each frame) and temporal (across frames) attention for efficiency

- Progressive synthesis: generates the intermediate frame progressively from coarse to fine,

with transformer attention applied at each resolution level

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VFIT(NeuralNetworkArchitecture<>,String,VFITOptions)` | Creates a VFIT model for ONNX inference. |
| `VFIT(NeuralNetworkArchitecture<>,VFITOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a VFIT model for native training and inference. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `Interpolate(Tensor<>,Tensor<>,Double)` |  |
| `PostprocessOutput(Tensor<>)` |  |
| `PreprocessFrames(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

