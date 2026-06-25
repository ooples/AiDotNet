---
title: "VFIMamba<T>"
description: "VFIMamba state-space model for video frame interpolation with linear-complexity attention."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.FrameInterpolation`

VFIMamba state-space model for video frame interpolation with linear-complexity attention.

## For Beginners

VFIMamba uses a new type of AI architecture called "Mamba" that can
process long sequences efficiently (unlike transformers which get slow with long inputs).
This means it can handle high-resolution frames without running out of memory, while still
understanding how every pixel relates to every other pixel.

## How It Works

**References:**

- Paper: "VFIMamba: Video Frame Interpolation with State Space Models" (Zhang et al., NeurIPS 2024)

VFIMamba is the first to apply Mamba (selective state space models) to frame interpolation:

- Selective state space: uses Mamba's selective scan mechanism instead of attention, achieving

linear complexity O(N) for processing frame features while maintaining global context,
enabling efficient processing of high-resolution frames

- Bidirectional scanning: scans frame features in both forward (left-to-right, top-to-bottom)

and backward directions, ensuring each pixel has full global context from all directions

- Cross-frame state propagation: the SSM state from one frame is propagated to condition

the processing of the other frame, enabling implicit motion correspondence without
explicit flow estimation

- Multi-scale Mamba blocks: hierarchical Mamba blocks at different spatial scales, capturing

both fine-grained texture details and coarse motion patterns

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VFIMamba(NeuralNetworkArchitecture<>,String,VFIMambaOptions)` | Creates a VFIMamba model for ONNX inference. |
| `VFIMamba(NeuralNetworkArchitecture<>,VFIMambaOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a VFIMamba model for native training and inference. |

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

