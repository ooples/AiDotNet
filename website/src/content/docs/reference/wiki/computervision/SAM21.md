---
title: "SAM21<T>"
description: "SAM 2.1: Segment Anything Model 2.1 with refined checkpoints for images and videos."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Foundation`

SAM 2.1: Segment Anything Model 2.1 with refined checkpoints for images and videos.

## For Beginners

SAM 2.1 is an updated version of SAM 2 with improved training recipes
that produce more accurate segmentation masks. Like SAM 2, it supports both image and video
segmentation through memory attention for temporal consistency.

Common use cases:

- Video object segmentation with click-to-track
- High-quality image segmentation with refined boundaries
- Interactive annotation tools
- Foundation for downstream video analysis

## How It Works

**Technical Details:**

- Hiera backbone (hierarchical vision transformer) — same as SAM 2
- Memory attention mechanism for multi-frame consistency
- Streaming architecture: processes frames sequentially
- Refined training recipes improve accuracy without architecture changes
- Supports 4 size variants: Tiny (39M), Small (46M), Base+ (81M), Large (224M)

**Reference:** Ravi et al., "SAM 2: Segment Anything in Images and Videos", Meta AI, 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SAM21(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,SAM21ModelSize,Double,SAM21Options)` | Initializes SAM 2.1 in native (trainable) mode. |
| `SAM21(NeuralNetworkArchitecture<>,String,Int32,SAM21ModelSize,SAM21Options)` | Initializes SAM 2.1 in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this SAM 2.1 instance supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddToMemory(Tensor<>)` | Adds image features to the memory bank for video tracking. |
| `ClearMemory` | Clears the memory bank. |
| `CreateNewInstance` | Creates a new instance with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads configuration from a binary stream. |
| `Dispose(Boolean)` | Releases managed resources. |
| `GetModelMetadata` | Collects metadata describing this model's configuration. |
| `GetOptions` | Gets the configuration options for this SAM 2.1 model. |
| `InitializeLayers` | Initializes the Hiera encoder and mask decoder layers. |
| `PredictCore(Tensor<>)` | Runs a forward pass through SAM 2.1. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes configuration to a binary stream. |
| `Train(Tensor<>,Tensor<>)` | Performs one training step. |
| `UpdateParameters(Vector<>)` | Updates all trainable parameters from a flat parameter vector. |

