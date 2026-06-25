---
title: "FastSAM<T>"
description: "FastSAM: Fast Segment Anything Model based on YOLOv8-Seg."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Efficient`

FastSAM: Fast Segment Anything Model based on YOLOv8-Seg.

## For Beginners

Real-time segment anything. Interactive segmentation on edge devices.

Common use cases:

- Real-time segment anything
- Interactive segmentation on edge devices
- Fast automatic mask generation
- Lightweight alternative to SAM

## How It Works

**Technical Details:**

- YOLOv8-Seg backbone trained on SA-1B dataset
- 50x faster than original SAM
- CNN-based architecture (no vision transformer)
- Supports point and box prompts

**Reference:** Zhao et al., "Fast Segment Anything", arXiv 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FastSAM(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Double,FastSAMOptions)` | Initializes FastSAM in native (trainable) mode. |
| `FastSAM(NeuralNetworkArchitecture<>,String,Int32,FastSAMOptions)` | Initializes FastSAM in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this FastSAM instance supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance with the same configuration but fresh weights. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads configuration from a binary stream. |
| `Dispose(Boolean)` | Releases managed resources including the ONNX inference session. |
| `GetModelMetadata` | Collects metadata describing this model's configuration. |
| `InitializeLayers` | Initializes the encoder and decoder layers. |
| `PredictCore(Tensor<>)` | Runs a forward pass to produce segmentation logits. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes configuration to a binary stream. |
| `Train(Tensor<>,Tensor<>)` | Performs one training step. |
| `UpdateParameters(Vector<>)` | Updates all trainable parameters from a flat parameter vector. |

