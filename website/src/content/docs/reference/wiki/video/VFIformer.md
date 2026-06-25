---
title: "VFIformer<T>"
description: "VFIformer cross-scale window transformer for video frame interpolation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.FrameInterpolation`

VFIformer cross-scale window transformer for video frame interpolation.

## For Beginners

VFIformer uses transformers (the same technology behind GPT and
other AI models) for frame interpolation. The attention mechanism lets every part of the
output frame "look at" relevant parts of both input frames, producing better results
especially for complex scenes.

## How It Works

**References:**

- Paper: "Video Frame Interpolation with Transformer" (Lu et al., CVPR 2022)

VFIformer applies vision transformers to frame interpolation with several key innovations:

- Cross-scale attention: transformer attention mechanism that attends across multiple feature

scales simultaneously, capturing both local fine-grained correspondences and global scene
structure in a single attention operation

- Flow-guided deformable attention: attention queries are positioned based on estimated

optical flow, so the model attends to motion-relevant regions rather than wasting attention
on irrelevant spatial locations

- Multi-frame transformer decoder: a transformer decoder that takes tokens from both input

frames and generates intermediate frame tokens, with causal masking adapted for spatial
rather than temporal ordering

- Efficient token design: uses feature pooling and stride patterns that reduce token count

by 16x compared to naive patch tokenization, enabling high-resolution processing

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VFIformer(NeuralNetworkArchitecture<>,String,VFIformerOptions)` | Creates a VFIformer model for ONNX inference. |
| `VFIformer(NeuralNetworkArchitecture<>,VFIformerOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a VFIformer model for native training and inference. |

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

