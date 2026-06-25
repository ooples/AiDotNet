---
title: "PixelLM<T>"
description: "PixelLM: Pixel Reasoning with Large Multimodal Model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Referring`

PixelLM: Pixel Reasoning with Large Multimodal Model.

## For Beginners

Multi-target reasoning segmentation. Complex text-to-mask segmentation.

Common use cases:

- Multi-target reasoning segmentation
- Complex text-to-mask segmentation
- Pixel-level reasoning from language
- Detailed visual grounding

## How It Works

**Technical Details:**

- Lightweight pixel decoder with codebook tokens
- Multi-scale feature integration for segmentation
- Handles multiple target masks from a single query
- End-to-end LLM + pixel decoder training

**Reference:** Ren et al., "PixelLM: Pixel Reasoning with Large Multimodal Model", CVPR 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PixelLM(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Double,PixelLMOptions)` | Initializes PixelLM in native (trainable) mode. |
| `PixelLM(NeuralNetworkArchitecture<>,String,Int32,PixelLMOptions)` | Initializes PixelLM in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this PixelLM instance supports training. |

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

