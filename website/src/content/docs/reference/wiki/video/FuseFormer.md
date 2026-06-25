---
title: "FuseFormer<T>"
description: "FuseFormer transformer-based video inpainting with fine-grained spatial-temporal fusion."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Inpainting`

FuseFormer transformer-based video inpainting with fine-grained spatial-temporal fusion.

## For Beginners

FuseFormer uses transformer attention to fuse information from multiple frames for video inpainting. It fills missing regions by attending to relevant visible content across the entire video.

## How It Works

**References:**

- Paper: "FuseFormer: Fusing Fine-Grained Information in Transformers for Video Inpainting" (Liu et al., ICCV 2021)

FuseFormer applies soft split and soft composition operations within a transformer encoder
to fuse fine-grained spatial-temporal features from overlapping patches, attending to both
local texture details and global structure across frames for high-quality inpainting.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FuseFormer(NeuralNetworkArchitecture<>,FuseFormerOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a FuseFormer model for native training and inference. |
| `FuseFormer(NeuralNetworkArchitecture<>,String,FuseFormerOptions)` | Creates a FuseFormer model for ONNX inference. |

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

