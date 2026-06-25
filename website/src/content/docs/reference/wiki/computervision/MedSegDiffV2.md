---
title: "MedSegDiffV2<T>"
description: "MedSegDiff-V2: Spectrum-space diffusion for medical segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Medical`

MedSegDiff-V2: Spectrum-space diffusion for medical segmentation.

## For Beginners

Medical image segmentation with diffusion models. High-quality boundary delineation.

Common use cases:

- Medical image segmentation with diffusion models
- High-quality boundary delineation
- Uncertain region handling in medical images
- Multi-scale medical feature extraction

## How It Works

**Technical Details:**

- Conditional diffusion process for segmentation mask generation
- Spectrum-Space Transformer (SS-Former) for multi-scale features
- Anchor condition + step-uncertainty-aware attention
- Iterative denoising for progressively refined segmentation

**Reference:** Wu et al., "MedSegDiff-V2: Diffusion-based Medical Image Segmentation with Transformer", AAAI 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MedSegDiffV2(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Double,MedSegDiffV2Options)` | Initializes MedSegDiffV2 in native (trainable) mode. |
| `MedSegDiffV2(NeuralNetworkArchitecture<>,String,Int32,MedSegDiffV2Options)` | Initializes MedSegDiffV2 in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this MedSegDiffV2 instance supports training. |

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

