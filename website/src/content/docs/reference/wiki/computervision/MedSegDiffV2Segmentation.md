---
title: "MedSegDiffV2Segmentation<T>"
description: "MedSegDiff-V2 Segmentation: Diffusion-based medical image segmentation pipeline."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Diffusion`

MedSegDiff-V2 Segmentation: Diffusion-based medical image segmentation pipeline.

## For Beginners

Diffusion-based medical segmentation. High-precision medical boundary delineation.

Common use cases:

- Diffusion-based medical segmentation
- High-precision medical boundary delineation
- Stochastic segmentation for uncertainty estimation
- Multi-modal medical image segmentation

## How It Works

**Technical Details:**

- Conditional diffusion model for segmentation mask denoising
- Spectrum-Space Former for frequency and spatial features
- Iterative refinement via reverse diffusion process
- Handles ambiguous medical image boundaries

**Reference:** Wu et al., "MedSegDiff-V2: Diffusion-based Medical Image Segmentation with Transformer", AAAI 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MedSegDiffV2Segmentation(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Double,MedSegDiffV2SegmentationOptions)` | Initializes MedSegDiffV2Segmentation in native (trainable) mode. |
| `MedSegDiffV2Segmentation(NeuralNetworkArchitecture<>,String,Int32,MedSegDiffV2SegmentationOptions)` | Initializes MedSegDiffV2Segmentation in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this MedSegDiffV2Segmentation instance supports training. |

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

